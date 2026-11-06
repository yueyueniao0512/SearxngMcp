import os
import re
import time
import logging
import threading
from concurrent.futures import (
    ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout
)
from urllib.parse import urlparse

import requests
import trafilatura
from mcp.server.fastmcp import FastMCP

# —— 日志：统一打到 stderr（不污染 stdio 的 JSON-RPC 协议流）——
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("searxng-local-fetch")

mcp = FastMCP("searxng-local-fetch")
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://127.0.0.1:8081")

# —— 复用 HTTP 会话：连接池 + 有限重试，避免每次请求新建连接、提升稳定性 ——
_SESSION = requests.Session()
_SESSION.mount(
    "http://", requests.adapters.HTTPAdapter(
        pool_connections=10, pool_maxsize=10,
        max_retries=requests.adapters.Retry(
            total=1, backoff_factor=0.3,
            status_forcelist=(500, 502, 503, 504),
            allowed_methods=frozenset(["GET"]),
        ),
    ),
)
_SESSION.mount(
    "https://", requests.adapters.HTTPAdapter(
        pool_connections=10, pool_maxsize=10,
        max_retries=requests.adapters.Retry(
            total=1, backoff_factor=0.3,
            status_forcelist=(500, 502, 503, 504),
            allowed_methods=frozenset(["GET"]),
        ),
    ),
)

# —— 共享抓取线程池：避免每次 fetch_page 新建并 shutdown 一个 executor ——
_FETCH_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="fetch")

# —— 并发安全：保护全局状态读写（stdio 下虽串行，留锁以防未来并发传输）——
_STATE_LOCK = threading.Lock()

# —— 防死循环：同一归一化查询在窗口内重复过多则熔断（按查询而非总次数）——
_QUERY_COUNTS = {}     # key -> [timestamps]
_LAST_RESULTS = {}     # key -> 该查询最近一次的真实返回，硬拒时回放给模型
_WINDOW_SECONDS = 300

# —— 主题级防死循环：换措辞也逃不掉——按核心词而非整句计数 ——
_TOPIC_COUNTS = {}     # 核心词 -> [timestamps]
_TOPIC_HISTORY = []    # (timestamp, key, terms, result)，熔断时回放该主题最近一次结果
_TOPIC_LIMIT = 3       # 同一核心词窗口内出现在 >3 次查询中 → 判定该主题未收敛
_STOP_TERMS = {"是什么", "什么", "怎么", "如何", "为什么", "介绍", "意思", "含义",
               "梗", "出处", "起源", "解释", "资料", "信息", "请问", "查询",
               "详细", "关于", "网络", "搜索"}

# —— fetch_page 防循环：客户端有“最大工具调用次数”上限，逐条抓取会先撞上去 ——
_FETCH_CACHE = {}      # url -> (text, timestamp)，重复抓取同一链接直接回放缓存
_FETCH_TIMES = []      # 真实抓取的时间戳（缓存命中不计次）
_FETCH_SOFT = 2        # 窗口内第 3 次真实抓取起，附加“尽快作答”提醒
_FETCH_HARD = 5        # 窗口内第 6 次起硬拒，劝模型基于已抓内容作答
_FETCH_FAILS = {}      # url -> 失败时间戳（负缓存：短时间内别重试同一个死链）
_FETCH_DEADLINE = float(os.getenv("FETCH_TIMEOUT", "25"))
_FETCH_MAX_CHARS = int(os.getenv("FETCH_MAX_CHARS", "3000"))
# 单页正文字符上限：每次 fetch_page 的返回都会整份进模型上下文，
# 多页叠加是 LM Studio 端 "failed to decode"（上下文溢出）的主要推手之一。
# 可用环境变量 FETCH_MAX_CHARS 调整。
# 单次抓取总时限（秒）：trafilatura 默认下载超时较长，会超过客户端对 MCP 调用
# 本身的超时（Cherry Studio 报 -32001），所以这里用更短的总时限，并给内部下载
# 留足但未超出，赶在客户端掐断之前返回明确提示。

# —— 已知登录墙/强反爬站点：抓取注定失败，直接提示跳过，省一次无效调用 ——
_BLOCKED_DOM = re.compile(
    r"(zhihu\.com|weixin\.qq\.com|xiaohongshu\.com|xhslink\.com|douyin\.com)")

_BROWSER_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/128.0.0.0 Safari/537.36"),
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}
_MAX_REPEAT = 3        # 第 3 次为“最后放行”（带强警告），第 4 次起硬拒并回放缓存

# —— 全局联合熔断：按“轮”计数，用户每发送一次消息视为新的一轮 ——
# 轮次判定：同一轮内工具调用是连续的（间隔几十秒），而两轮之间必然存在
# 用户阅读+输入的静默期。距上次工具调用静默超过 _ROUND_GAP 秒即视为
# 新一轮开始，所有熔断计数清零——上一轮的额度不会影响用户下一条消息。
# 可用环境变量 ROUND_GAP 调整静默阈值（秒）。
_ROUND_CALLS = []      # 本轮已发生的真实工具调用时间戳
_ROUND_GAP = float(os.getenv("ROUND_GAP", "120"))  # 静默多久算新一轮（秒）
_ROUND_LIMIT = 10      # 一轮内真实调用达到 10 次 → 两个工具一并全面熔断
_LAST_TOOL_TS = [None] # 最近一次工具调用时间（单调时钟），用于轮次判定

# —— 全局状态上限，防止长周期运行内存无限增长 ——
_CAP_QUERY = 500
_CAP_TOPIC = 500
_CAP_CACHE = 200
_CAP_LAST = 200

# —— 通用字典噪音模式（标题命中即降权，非针对特定查询）——
_DICT_NOISE = re.compile(r"(的意思|的解释|的拼音|的笔顺|的部首|汉语文字|汉语字典|汉典|汉语国学|词典)")

# —— 字典型提问（单字/词 + 释义类词）：命中则不剥字典结果 ——
_DICT_QUERY = re.compile(r"(的意思|什么意思|怎么读|拼音|笔顺|部首)$")

# —— 英文虚词停用表：不参与核心词提取（否则 "the" 等会污染相关性计数）——
_EN_STOP = {"the", "a", "an", "of", "and", "or", "to", "in", "on", "at", "for",
            "with", "is", "are", "was", "were", "be", "been", "what", "how",
            "why", "when", "who", "new", "is"}


def _cap_dict(d, max_size):
    """按插入顺序淘汰最旧条目，避免字典无限增长（配合下面的状态字典使用）。"""
    while len(d) > max_size:
        d.pop(next(iter(d)))


def _global_guard(now):
    """轮次判定 + 全局联合熔断。

    轮次定义：用户每发送一次消息为一轮。两次工具调用之间静默超过
    _ROUND_GAP 秒即视为新一轮开始，清空全部熔断计数（查询级/主题级/
    抓取级/全局），保证上一轮的额度不会影响用户下一条消息。
    随后按“本轮合计”做最后闸门：搜索+抓取合计达到 _ROUND_LIMIT 即全面
    拒绝——即使模型换措辞、在两个工具间交替循环也逃不掉。
    返回 True 表示本轮已超限。
    """
    with _STATE_LOCK:
        last = _LAST_TOOL_TS[0]
        if last is None or now - last > _ROUND_GAP:
            _ROUND_CALLS.clear()
            _QUERY_COUNTS.clear()
            _TOPIC_COUNTS.clear()
            _TOPIC_HISTORY.clear()
            _FETCH_TIMES.clear()
        _LAST_TOOL_TS[0] = now
        if len(_ROUND_CALLS) >= _ROUND_LIMIT:
            return True
        _ROUND_CALLS.append(now)
    return False


_GLOBAL_REJECT_MSG = (
    f"⚠️ 本轮任务的工具调用（搜索+抓取合计）已达 {int(_ROUND_LIMIT)} 次上限，"
    "所有工具现已停止响应。请立即停止调用任何工具，"
    "直接基于已获取的信息或你自己的知识回答用户；"
    "若信息不足以回答，请如实告知用户信息有限，并注明未经联网核实。")


def _record_history(now, key, guard_terms, out):
    """统一记录“最近结果”与“主题历史”（原多处重复逻辑收敛为一处）。"""
    _LAST_RESULTS[key] = out
    _TOPIC_HISTORY.append((now, key, frozenset(guard_terms), out))
    if len(_TOPIC_HISTORY) > 60:
        del _TOPIC_HISTORY[:-60]
    _cap_dict(_LAST_RESULTS, _CAP_LAST)


def _lang_for(query: str):
    """含中日韩字符时显式指定中文，否则不强制语言（通用方案，不写死 zh）。"""
    return "zh-CN" if re.search(r"[一-鿿]", query) else None


def _core_terms(query: str):
    """抽取查询核心词：去掉 site:、疑问前缀与过短/停用词，作为相关性判定依据。

    版本号类 token（如 1.21）中的点号不拆分，避免裂成 "1"/"21" 噪音词。
    中文无空格，"什么是原神" 这类疑问前缀会粘住主题词，需先剥离。
    """
    q = re.sub(r"site:\S+", " ", query, flags=re.IGNORECASE)
    q = re.sub(r"^(请问|什么是|什么是|介绍一下|帮我|帮我找|搜索|查一下|查查|看看)+", " ", q)
    q = re.sub(r"[^\w一-鿿.]+", " ", q)
    out = []
    for t in q.split():
        t = t.strip(".").lower()
        if len(t) >= 2 and t not in _EN_STOP:
            out.append(t)
    return out


def _term_weight(t):
    """词的区分度权重：越长的词越具体；CJK 窄词（如“新维度”）再加成。"""
    return len(t) + (2 if re.search(r"[一-鿿]", t) else 0)


def _narrow_term(terms):
    """挑单个最具区分度的窄词：混排/中文查询取最先出现的 CJK 限定词，
    纯英文/无 CJK 时取权重最高词。

    依据：用户输入顺序反映重要性——"minecraft 新维度 自家更新" 里
    "新维度" 才是核心限定，且它最先出现。绝不默认选首个泛词
    （如 "minecraft"）——单词搜索用泛词只会召回海量官网噪音，
    把真正的窄词结果淹没（正是旧版截图里的问题）。
    """
    if not terms:
        return ""
    cjk = [t for t in terms if re.search(r"[一-鿿]", t)]
    if cjk:
        return cjk[0]
    return max(terms, key=_term_weight)


def _top_combo(terms, n=2):
    """组合查询词：主题头词（用户放在最前的词，如 minecraft）
    + 最先出现的 CJK 限定词（如 新维度）。

    用户查 "minecraft 新维度" 时想搜的就是这两个词连起来的整体——
    组合搜索理应自动完成这件事；拆成单个词只作为最后的补充手段。
    """
    if not terms:
        return ""
    if n <= 1 or len(terms) == 1:
        return _narrow_term(terms)
    head = terms[0]
    narrow = _narrow_term(terms)
    if narrow != head:
        return f"{head} {narrow}"
    rest = sorted((t for t in terms[1:] if t != head),
                  key=_term_weight, reverse=True)
    return " ".join([head] + rest[:n - 1])


def _score(item, terms):
    """结果与查询的匹配度：按“命中了哪些词”加权计分（区分度高的词权重大）。

    旧的 `命中词个数` 计分会把只含 "minecraft" 的离题官网判为相关；
    加权后 "minecraft + 新维度" 双命中远高于 "minecraft" 单命中，
    排序时真正的组合相关结果才能浮到前面。
    """
    blob = (item.get("title", "") + " " + item.get("content", "")).lower()
    return sum(_term_weight(t) for t in terms if t.lower() in blob)


def _is_relevant(item, terms):
    """判定单条结果是否“真正相关”（多词查询的门槛远高于旧版 `_score > 0`）。

    - 多词查询：必须命中 ≥2 个不同核心词；只命中 1 个词时，仅当该词是
      CJK 窄词（混排查询里真正的限定词，如“新维度”）才算相关——
      防止 "minecraft" 这类高频泛词单命中就把离题结果混进来。
    - 单词查询：命中该词即可。
    """
    if not terms:
        return True
    blob = (item.get("title", "") + " " + item.get("content", "")).lower()
    matched = [t for t in terms if t.lower() in blob]
    if len(matched) >= 2:
        return True
    if len(terms) == 1:
        return len(matched) == 1
    return len(matched) == 1 and re.search(r"[一-鿿]", matched[0]) is not None


def _is_dict_lookup(query: str):
    """判断用户是否在查字/词释义（这类查询应保留字典结果）。"""
    q = query.strip()
    if re.search(_DICT_QUERY, q):
        return True
    core = re.sub(r"\s+", "", q)
    return len(core) <= 2 and re.search(r"[一-鿿]", core) is not None


def _broaden(query: str):
    """稀疏时扩词：去掉英文修饰，只留中文核心词；无中文则取最长词。"""
    terms = _core_terms(query)
    cjk = [t for t in terms if re.search(r"[一-鿿]", t)]
    if cjk:
        return " ".join(cjk[:3])
    return max(terms, key=len) if terms else query


def _fetch_results(query, k=10):
    """调用本地 SearXNG（JSON 格式）。失败仅记日志并返回空列表，不抛异常。"""
    params = {"q": query, "format": "json"}
    lang = _lang_for(query)
    if lang:
        params["language"] = lang
    try:
        resp = _SESSION.get(f"{SEARXNG_URL}/search", params=params, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning("SearXNG 搜索请求失败 q=%r: %s", query, e)
        return []
    try:
        return resp.json().get("results", [])[:k]
    except ValueError:
        logger.warning("SearXNG 返回非 JSON（可能服务异常）q=%r", query)
        return []


def _format(items):
    blocks = []
    for it in items:
        title = it.get("title", "")
        url = it.get("url", "")
        content = (it.get("content", "") or "")[:300]
        blocks.append(f"【{title}】\n{url}\n{content}\n")
    return "\n---\n".join(blocks)


def _diverse_top(items, terms, n=10):
    """按相关性排序后，每域名至多 1 条优先铺开，保证结果多样性与覆盖。

    用 URL 集合去重（原实现用 `it not in picked` 做字典全等比较，既慢又可能漏去重）。
    """
    items = sorted(items, key=lambda it: _score(it, terms), reverse=True)
    picked, seen_dom, seen_url = [], set(), set()
    for it in items:
        dom = urlparse(it.get("url", "")).netloc
        u = it.get("url", "")
        if dom not in seen_dom and u not in seen_url:
            seen_dom.add(dom)
            seen_url.add(u)
            picked.append(it)
        if len(picked) >= n:
            break
    if len(picked) < n:
        for it in items:
            u = it.get("url", "")
            if u not in seen_url:
                seen_url.add(u)
                picked.append(it)
            if len(picked) >= n:
                break
    return picked[:n]


@mcp.tool()
def local_search(query: str) -> str:
    """本地 SearXNG 网页搜索（整体优先的组合搜索 + 严格相关性过滤 + 防空转）

    搜索策略（自动完成，无需拆词重试）：
    1. 多关键词会先作为整体组合直接搜索（引擎自己做 AND 式加权排序）；
    2. 命中不足时自动尝试：去噪全组合 → 精确短语 → 最具区分度的
       2 词组合（如 "minecraft 新维度"）→ 单个窄词（最后的补充）；
    3. 相关性门槛：多词查询的结果必须命中 ≥2 个核心词才算相关，
       只命中首个泛词（如仅含 "minecraft"）的离题结果会被剔除。

    自适应三档：
    - 相关结果多（大众话题）：严格去噪 + 每域名至多 1 条铺开 + 返回前 10 条
    - 相关结果少（小众话题）：宽松保准，命中核心词的几条全给
    - 零命中：自动扩词重试；仍无则给带警示的“尽力样本”，绝不返回纯字典垃圾

    使用约定：同一主题最多搜索 2~3 轮；换措辞、换近义词的反复重试同样会被计数并熔断。
    若已搜索多轮仍无满意结果，请直接基于已有信息作答，不要再调用本工具。

    Args:
        query: 搜索关键词（可含 site: 等运算符，多关键词用空格分隔即可）
    """
    now = time.monotonic()
    key = re.sub(r"[\s\W_]+", "", query).lower()

    # —— 全局联合熔断（最后闸门）：搜索+抓取合计超限，立即全面停手 ——
    if _global_guard(now):
        return _GLOBAL_REJECT_MSG

    # —— 防死循环（查询级）：1~2 次正常；第 3 次最后放行；第 4 次起硬拒并回放缓存 ——
    with _STATE_LOCK:
        ts = _QUERY_COUNTS.setdefault(key, [])
        ts.append(now)
        ts[:] = [t for t in ts if now - t <= _WINDOW_SECONDS]
        if not ts:
            del _QUERY_COUNTS[key]
        _cap_dict(_QUERY_COUNTS, _CAP_QUERY)
        repeated = len(ts) > _MAX_REPEAT
        cached = _LAST_RESULTS.get(key) if repeated else None

    if repeated:
        if cached:
            return ("⚠️ 该查询已重复搜索多次，不再执行新搜索。"
                    "以下为此前最后一次的搜索结果，请直接基于它回答用户，"
                    "不要再调用搜索工具。\n\n" + cached)
        return ("⚠️ 同一查询已重复搜索多次仍未收敛，请立即停止调用搜索工具。"
                "请基于已获取的信息或你自己的知识直接回答用户，并注明信息可能未经联网核实。")

    # —— 主题级熔断：整句换了措辞没用，核心词反复出现同样算未收敛 ——
    guard_terms = [t for t in _core_terms(query) if t not in _STOP_TERMS]
    trip_term = None
    with _STATE_LOCK:
        for t in set(guard_terms):
            tss = _TOPIC_COUNTS.setdefault(t, [])
            tss.append(now)
            tss[:] = [x for x in tss if now - x <= _WINDOW_SECONDS]
            if not tss:
                del _TOPIC_COUNTS[t]
            if len(tss) > _TOPIC_LIMIT and trip_term is None:
                trip_term = t
        _cap_dict(_TOPIC_COUNTS, _CAP_TOPIC)
    if trip_term:
        with _STATE_LOCK:
            latest = next((e for e in reversed(_TOPIC_HISTORY)
                           if trip_term in e[2]), None)
        if latest:
            return (f"⚠️ 关于“{trip_term}”的搜索已重复多轮仍未收敛，不再执行新搜索。"
                    "请立即停止调用搜索工具，直接基于以下最近一次结果回答用户；"
                    "若仍不足以回答，请如实告知用户信息有限，不要继续换词搜索。\n\n"
                    + latest[3])
        return (f"⚠️ 关于“{trip_term}”的搜索已重复多轮仍未收敛，请立即停止调用搜索工具。"
                "请基于已获取的信息或你自己的知识直接回答用户，并注明信息可能未经联网核实。")

    has_site = "site:" in query.lower()
    bare = re.sub(r"site:\S+", " ", query, flags=re.IGNORECASE)
    bare = re.sub(r"\s+", " ", bare).strip()
    terms = _core_terms(bare or query)

    # —— 搜索策略（核心原则：整体优先，组合次之，拆词只做最后补充）——
    # ① 用户原始查询原样发给引擎（含 site: 时）：多关键词交给引擎做
    #    AND 式加权排序，这是对 "minecraft 1.21 新维度" 最准确的召回；
    # ② 去噪核心词全组合（等价于原查询但剔除了口语噪音词）；
    # ③ 精确短语（仅当原文措辞确实逐字存在时才命中，作精确兜底，不再当首选）；
    # ④ 最具区分度的 2 个核心词组合（如 "minecraft 新维度"——泛词+窄词
    #    连起来搜，这正是用户想查的组合）；
    # ⑤ 拆成单个窄词（最后的补充手段，绝不作为常规回退）。
    attempts = []
    if has_site:
        attempts.append(query.strip())
    if bare:
        attempts.append(bare)
    if terms:
        full_combo = " ".join(terms)
        attempts.append(full_combo)
        attempts.append(f'"{bare}"')
        combo = _top_combo(terms, 2)
        if combo and combo != full_combo:
            attempts.append(combo)
    seen_a, uniq = set(), []
    for a in attempts:
        if a and a not in seen_a:
            seen_a.add(a)
            uniq.append(a)

    def _grab(q):
        return _fetch_results(q)

    # 第 1 步：只发完整组合（整体优先），命中足够就直接返回，多数查询到此为止
    seen, collected = set(), []

    def _merge(items):
        for it in items:
            u = it.get("url", "")
            if u and u not in seen:
                seen.add(u)
                collected.append(it)

    if uniq:
        _merge(_grab(uniq[0]))
    if sum(1 for it in collected if _is_relevant(it, terms)) >= 8:
        uniq_rest = []
    else:
        uniq_rest = uniq[1:]

    # 第 2 步：整体策略命中不足，才并行发起其余组合变体（墙钟 = 最慢一次）
    if uniq_rest:
        with ThreadPoolExecutor(max_workers=min(4, len(uniq_rest))) as ex:
            for fut in as_completed([ex.submit(_grab, q) for q in uniq_rest]):
                _merge(fut.result())

    # 第 3 步：组合全稀疏时，才拆词补充——只搜最具区分度的窄词（而非首个泛词），
    # 避免 "minecraft" 单词搜索用海量官网噪音淹没关系词结果
    narrow_term = _narrow_term(terms)
    if (sum(1 for it in collected if _is_relevant(it, terms)) < 5
            and narrow_term and narrow_term not in seen_a):
        _merge(_grab(narrow_term))
        seen_a.add(narrow_term)

    dict_lookup = _is_dict_lookup(bare or query)
    # 相关性过滤：多词查询要求命中 ≥2 个核心词（或 CJK 窄词单命中），
    # 不再是旧版的 "命中任意 1 个词就算相关"
    good = [it for it in collected if _is_relevant(it, terms)]
    if not dict_lookup:
        good = [it for it in good if not _DICT_NOISE.search(it.get("title", ""))]

    if len(good) >= 5:
        # 第 1 档：大众，精选 + 多样性 + 给全
        out = (_format(_diverse_top(good, terms, n=10))
               + "\n（如需更多细节，请从以上结果中挑选最相关的 2~3 条链接"
                 "调用 fetch_page 阅读正文，不要逐条抓取全部链接；"
                 "多数问题基于标题和摘要即可回答）")
        with _STATE_LOCK:
            _record_history(now, key, guard_terms, out)
        return out

    if len(good) >= 1:
        # 第 2 档：小众，全给（命中核心词即保留，不误杀稀疏命中）
        out = _format(good[:10])
        if len(good) < 3:
            out += ("\n（相关结果较少：若不足以回答，请勿用近似关键词反复重试，"
                    "可换更通用的关键词或基于已有知识作答）")
        with _STATE_LOCK:
            _record_history(now, key, guard_terms, out)
        return out

    # 第 3 档：零命中 → 扩词重试，再给带警示的“尽力样本”
    broad = _broaden(bare or query)
    if broad and broad not in seen_a:
        for it in _fetch_results(broad):
            u = it.get("url", "")
            if u and u not in seen:
                seen.add(u)
                collected.append(it)
    pool = [it for it in collected
            if dict_lookup or not _DICT_NOISE.search(it.get("title", ""))]
    pool = sorted(pool, key=lambda it: _score(it, terms), reverse=True)
    best = pool[:3]
    if best:
        out = (_format(best)
               + "\n⚠️ 未匹配到与查询高度相关的结果，以上仅供参考，可能离题。"
                 "请勿再调用搜索工具，直接基于以上信息或已有知识作答。")
        with _STATE_LOCK:
            _record_history(now, key, guard_terms, out)
        return out
    return ("未找到与查询相关的结果（已尝试精确短语、裸词、关键词及扩词检索）。\n"
            "请勿用相近关键词重复搜索；建议换用完全不同的关键词，"
            "或直接基于已有知识回答用户并说明未能联网核实。")


@mcp.tool()
def fetch_page(url: str) -> str:
    """抓取目标网页的完整正文，供模型基于原文作答（而非搜索摘要）。

    使用约定：只抓取与回答直接相关的少量链接（最多 2~3 条，选最相关的），
    不要逐条抓取搜索结果的全部链接；不要重复抓取同一 URL（有缓存回放）。

    Args:
        url: 目标网页链接（原始地址，非搜索摘要）
    """
    now = time.monotonic()
    norm = url.strip()

    # —— 全局联合熔断（最后闸门）：搜索+抓取合计超限，立即全面停手 ——
    if _global_guard(now):
        return _GLOBAL_REJECT_MSG

    # 同一 URL 窗口内重复抓取：直接回放缓存，不计次、不重新下载
    cached = _FETCH_CACHE.get(norm)
    if cached and now - cached[1] <= _WINDOW_SECONDS:
        return ("（⚠️ 此页面已抓取过，以下为缓存内容，请勿再次抓取同一链接）\n\n"
                + cached[0])

    # 窗口内真实抓取计数：达到硬上限则拒绝，劝模型基于已抓内容作答
    with _STATE_LOCK:
        _FETCH_TIMES[:] = [t for t in _FETCH_TIMES if now - t <= _WINDOW_SECONDS]
        if len(_FETCH_TIMES) >= _FETCH_HARD:
            return ("⚠️ 本轮任务已抓取过多页面，不再执行新的抓取。"
                    "请立即停止调用 fetch_page，基于已获取的内容直接回答用户。")
        _FETCH_TIMES.append(now)

        # 已知登录墙/强反爬站点：直接告知跳过，不要浪费调用去撞墙
        dom = urlparse(norm).netloc
        if _BLOCKED_DOM.search(dom):
            return (f"⚠️ {dom} 为登录墙/强反爬站点，工具无法抓取其正文。"
                    "请勿重试此 URL 或其变体，改选搜索结果中的其他来源，"
                    "或直接基于搜索结果中的摘要作答。")

        # 负缓存：刚才抓失败的 URL，短时间内别再撞第二次
        fail_ts = _FETCH_FAILS.get(norm)
        if fail_ts and now - fail_ts <= 120:
            return ("⚠️ 此 URL 刚刚抓取失败（该站点可能反爬或需登录），已跳过重试。"
                    "请改选其他来源，或基于搜索结果中的摘要作答。")

    # 内部下载超时略小于外层总时限，给“trafilatura 主路径失败 → 浏览器兜底”留余量，
    # 避免出现“外层已掐断、后台线程仍在慢速下载”的悬挂。
    fetch_timeout = max(8, int(_FETCH_DEADLINE) - 12)

    def _job():
        try:
            # 主路径：trafilatura 一站式抓取 + 抽正文
            downloaded = trafilatura.fetch_url(norm, timeout=fetch_timeout)
            text = (trafilatura.extract(downloaded, include_comments=False)
                    if downloaded else None)
            # 兜底：仅当 trafilatura 自身下载被拒（常见于默认 UA 被识别）时，
            # 换浏览器身份重试一次；正文抽取仍交给 trafilatura
            if not text:
                resp = _SESSION.get(norm, headers=_BROWSER_HEADERS, timeout=10)
                if resp.ok and resp.text:
                    text = trafilatura.extract(resp.text, include_comments=False)
            return text
        except Exception as e:
            logger.warning("抓取失败 %s: %s", norm, e)
            return None

    # 硬时限：整体跑在共享线程池里，超时立即返回明确提示，
    # 决不允许单个慢站点把 MCP 调用拖到客户端超时（-32001）
    try:
        text = _FETCH_EXECUTOR.submit(_job).result(timeout=_FETCH_DEADLINE)
    except FuturesTimeout:
        with _STATE_LOCK:
            _FETCH_FAILS[norm] = now
        logger.warning("抓取超时 %s（>%ss）", norm, _FETCH_DEADLINE)
        return (f"页面加载超时（超过 {_FETCH_DEADLINE:.0f} 秒已中止），"
                "该站点可能响应缓慢或拦截抓取。请勿重试此 URL，"
                "改选其他来源，或基于搜索结果中的摘要作答。")

    if not text:
        with _STATE_LOCK:
            _FETCH_FAILS[norm] = now
        return ("页面无法提取正文（可能连接超时、被拦截或为纯脚本/登录墙页面）。"
                "请勿重试此 URL，改选其他来源，或基于搜索结果中的摘要作答。")

    body = text[:_FETCH_MAX_CHARS]
    with _STATE_LOCK:
        if len(_FETCH_TIMES) > _FETCH_SOFT:
            body += ("\n\n（⚠️ 已连续抓取多个页面，请尽快基于已获取的信息作答，"
                     "不要再抓取更多链接）")
        _FETCH_CACHE[norm] = (body, now)
        _cap_dict(_FETCH_CACHE, _CAP_CACHE)
    return body


if __name__ == "__main__":
    mcp.run()
