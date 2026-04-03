"""
VimRAG Demo: 多模态记忆图可视化 Agent Demo
一个美观的 Streamlit 前端，展示 VimRAG Agent 的 DAG 推理过程
"""

import streamlit as st
import streamlit.components.v1 as components
import json
import json5
import re
import os
import html as html_lib
from time import sleep

from vimrag_agent import VimRAG

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="VimRAG: Visual Memory Graph Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 自定义 CSS 样式 ====================
def inject_custom_css():
    """注入自定义 CSS 样式 - Kimi 风格明亮浅色主题"""
    st.markdown("""
    <style>
    /* 整体页面明亮背景 */
    .stApp {
        background: #f7f8fa;
    }
    
    /* 隐藏默认 Streamlit 元素 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* 侧边栏样式 - 浅色 */
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e5e7eb;
    }
    
    /* 侧边栏标题 - 蓝色调 */
    .sidebar-title {
        font-size: 2rem;
        font-weight: 700;
        color: #4f6ef7;
        margin-bottom: 0.25rem;
    }
    
    .sidebar-subtitle {
        font-size: 0.85rem;
        color: #999;
        margin-bottom: 1.5rem;
    }
    
    /* 配置区域标题 */
    .config-header {
        font-size: 0.9rem;
        color: #4f6ef7;
        font-weight: 600;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    

    
    /* 主内容区标题 */
    .main-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1a1a2e;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .main-header .icon {
        font-size: 1.5rem;
    }
    
    /* 思考区域 - 浅灰背景蓝色左边框 */
    .think-block {
        border-left: 3px solid #4f6ef7;
        padding: 12px 16px;
        margin: 10px 0;
        background: #f5f6f8;
        border-radius: 0 10px 10px 0;
        font-style: italic;
        color: #888;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    .think-label {
        color: #4f6ef7;
        font-weight: 600;
        font-style: normal;
        cursor: pointer;
        display: list-item;
    }

    .think-content {
        margin-top: 6px;
        white-space: pre-wrap;
        word-break: break-word;
    }
    
    /* 搜索徽章 - 白色背景蓝色边框 */
    .search-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: #ffffff;
        color: #4f6ef7;
        padding: 8px 18px;
        border-radius: 25px;
        font-weight: 600;
        margin: 10px 0;
        border: 1px solid #4f6ef7;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
        font-size: 0.95rem;
    }
    
    .search-badge .icon {
        font-size: 1.1rem;
    }
    
    .search-query {
        color: #4f6ef7;
        font-weight: 500;
    }
    
    /* 文件标签容器 - 浅灰背景 */
    .files-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin: 10px 0;
        padding: 12px;
        background: #fafbfc;
        border-radius: 10px;
        border: 1px solid #e5e7eb;
    }
    
    .files-label {
        width: 100%;
        color: #666;
        font-size: 0.8rem;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    
    /* 文件标签 - 浅灰标签 */
    .file-tag {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background: #f0f2f5;
        border: 1px solid #e5e7eb;
        padding: 5px 12px;
        border-radius: 15px;
        font-size: 0.82rem;
        color: #666;
        transition: all 0.2s ease;
    }
    
    .file-tag:hover {
        background: #e5e7eb;
        border-color: #d1d5db;
    }
    
    .file-tag .type-icon {
        font-size: 0.9rem;
    }
    
    /* 记忆卡片 - 白色背景绿色左边框 */
    .memory-card {
        background: #ffffff;
        border-left: 4px solid #10b981;
        border-radius: 12px;
        padding: 14px 18px;
        margin: 12px 0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
    }
    
    .memory-header {
        display: flex;
        align-items: center;
        gap: 8px;
        color: #10b981;
        font-weight: 600;
        margin-bottom: 8px;
        font-size: 0.9rem;
    }
    
    .memory-content {
        color: #333;
        font-size: 0.88rem;
        line-height: 1.6;
    }
    
    /* 答案卡片 - 白色背景蓝色顶部边框 */
    .answer-card {
        background: #ffffff;
        border-top: 3px solid #4f6ef7;
        border-radius: 16px;
        padding: 20px 24px;
        margin: 20px 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }
    
    .answer-header {
        display: flex;
        align-items: center;
        gap: 10px;
        color: #4f6ef7;
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 12px;
    }
    
    .answer-content {
        color: #333;
        font-size: 1rem;
        line-height: 1.8;
    }
    
    /* 错误卡片 - 浅红背景红左边框 */
    .error-card {
        background: #fef2f2;
        border-left: 4px solid #ef4444;
        border-radius: 10px;
        padding: 12px 16px;
        margin: 10px 0;
        color: #dc2626;
    }
    
    /* 图区域标题 - 蓝色 */
    .graph-header {
        display: flex;
        align-items: center;
        gap: 10px;
        color: #4f6ef7;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1rem;
        padding-bottom: 10px;
        border-bottom: 1px solid #e5e7eb;
    }
    
    /* 图容器 - 白色卡片 */
    .graph-container {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 15px;
        min-height: 450px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
    }
    
    /* 步骤指示器 - 蓝色 */
    .step-indicator {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 26px;
        height: 26px;
        background: #4f6ef7;
        border-radius: 50%;
        color: #fff;
        font-size: 0.75rem;
        font-weight: 700;
        margin-right: 10px;
        box-shadow: 0 2px 8px rgba(79, 110, 247, 0.3);
    }
    
    /* 分割线 - 浅灰渐变 */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #e5e7eb, transparent);
        margin: 20px 0;
    }
    
    /* 输入框区域 - 固定在顶部 */
    .input-section {
        background: #f7f8fa;
        border: none;
        border-radius: 0;
        padding: 15px 20px;
        margin-top: 0;
        box-shadow: none;
        position: sticky;
        top: 0;
        z-index: 999;
        border-bottom: 1px solid #e5e7eb;
    }
    
    .input-label {
        color: #4f6ef7;
        font-weight: 600;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Streamlit 输入框样式覆盖 - 白色背景浅灰边框 */
    .stTextInput > div > div > input {
        background: #ffffff !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 10px !important;
        color: #333 !important;
        padding: 12px 16px !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #4f6ef7 !important;
        box-shadow: 0 0 0 2px rgba(79, 110, 247, 0.1) !important;
    }
    
    /* 按钮样式 - 蓝色主色调 */
    .stButton > button {
        background: #4f6ef7 !important;
        color: #fff !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 30px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 8px rgba(79, 110, 247, 0.3) !important;
    }
    
    .stButton > button:hover {
        background: #3b5bdb !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(79, 110, 247, 0.4) !important;
    }
    
    /* 状态指示 - 蓝色 */
    .status-running {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        color: #4f6ef7;
        font-size: 0.9rem;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* 空状态提示 - 浅色风格 */
    .empty-state {
        text-align: center;
        color: #999;
        padding: 60px 20px;
        font-size: 1rem;
        background: #fafbfc;
        border-radius: 12px;
        border: 1px dashed #e5e7eb;
    }
    
    .empty-state .icon {
        font-size: 3rem;
        margin-bottom: 15px;
        opacity: 0.5;
    }
    
    /* 滚动条样式 - 浅色风格 */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f0f2f5;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #d1d5db;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #9ca3af;
    }
    
    /* ==================== 布局固定：右侧 Graph sticky + 左侧滚动 ==================== */
    
    /* 主内容区让两列形成固定布局 */
    [data-testid="stHorizontalBlock"] {
        align-items: flex-start !important;
    }
    
    /* 右列容器固定在视口顶部 */
    [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(2) > div {
        position: sticky;
        top: 1rem;
        max-height: calc(100vh - 2rem);
        overflow-y: auto;
    }
    
    /* 左列滚动区域 */
    [data-testid="stHorizontalBlock"] > [data-testid="column"]:first-child > div {
        position: sticky;
        top: 1rem;
        max-height: calc(100vh - 2rem);
        overflow-y: auto;
    }
    
    /* 右列白色背景卡片 */
    [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(2) > div > div {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 15px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
    }
    </style>
    """, unsafe_allow_html=True)


# ==================== vis.js 图渲染函数 ====================
def render_graph_html(nodes, edges, height=480):
    """
    生成 vis.js 图的 HTML 代码
    
    Args:
        nodes: [{"id": str, "label": str, "title": str, "color": dict, "shape": str, ...}]
        edges: [{"from": str, "to": str}]
        height: 图的高度（像素）
    
    Returns:
        完整的 HTML 字符串
    """
    nodes_json = json.dumps(nodes, ensure_ascii=False)
    edges_json = json.dumps(edges, ensure_ascii=False)
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            body {{
                background: transparent;
                overflow: hidden;
            }}
            #graph {{
                width: 100%;
                height: {height}px;
                background: transparent;
            }}
        </style>
    </head>
    <body>
        <div id="graph"></div>
        <script>
            var nodes = new vis.DataSet({nodes_json});
            var edges = new vis.DataSet({edges_json});
            var container = document.getElementById('graph');
            
            var options = {{
                layout: {{
                    hierarchical: {{
                        enabled: true,
                        direction: 'UD',
                        sortMethod: 'directed',
                        levelSeparation: 100,
                        nodeSpacing: 180,
                        treeSpacing: 200,
                        blockShifting: true,
                        edgeMinimization: true,
                        parentCentralization: true
                    }}
                }},
                nodes: {{
                    font: {{
                        color: '#333333',
                        size: 13,
                        face: 'Arial, sans-serif',
                        strokeWidth: 2,
                        strokeColor: '#ffffff'
                    }},
                    borderWidth: 2,
                    shadow: {{
                        enabled: true,
                        color: 'rgba(0, 0, 0, 0.1)',
                        size: 8,
                        x: 0,
                        y: 2
                    }},
                    scaling: {{
                        label: {{
                            enabled: true,
                            min: 11,
                            max: 15
                        }}
                    }}
                }},
                edges: {{
                    color: {{
                        color: '#cbd5e1',
                        highlight: '#6366f1',
                        hover: '#6366f1'
                    }},
                    width: 2,
                    arrows: {{
                        to: {{
                            enabled: true,
                            scaleFactor: 0.7,
                            type: 'arrow'
                        }}
                    }},
                    smooth: {{
                        enabled: true,
                        type: 'cubicBezier',
                        forceDirection: 'vertical',
                        roundness: 0.4
                    }},
                    shadow: {{
                        enabled: true,
                        color: 'rgba(0, 0, 0, 0.08)',
                        size: 3
                    }}
                }},
                physics: {{
                    enabled: false
                }},
                interaction: {{
                    hover: true,
                    tooltipDelay: 150,
                    zoomView: true,
                    dragView: true,
                    dragNodes: false
                }}
            }};
            
            var network = new vis.Network(container, {{nodes: nodes, edges: edges}}, options);
            
            // 自动适应视图
            network.once('stabilized', function() {{
                network.fit({{
                    animation: {{
                        duration: 500,
                        easingFunction: 'easeInOutQuad'
                    }}
                }});
            }});
        </script>
    </body>
    </html>
    """
    return html


# ==================== 辅助函数 ====================
def init_session_state():
    """初始化 session state"""
    defaults = {
        'vis_nodes': [],
        'vis_edges': [],
        'process_log': [],
        'is_running': False,
        'current_query': '',
        'last_node_id': None,
        'step_count': 0
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_state():
    """重置状态（新问题时调用）"""
    st.session_state.vis_nodes = []
    st.session_state.vis_edges = []
    st.session_state.process_log = []
    st.session_state.is_running = False
    st.session_state.last_node_id = None
    st.session_state.step_count = 0


def add_root_node(query):
    """添加根节点"""
    st.session_state.vis_nodes = [{
        "id": "root",
        "label": "🌐 Query",
        "title": f"用户问题: {query}",
        "color": {"background": "#6366f1", "border": "#4f46e5"},
        "shape": "circle",
        "size": 38,
        "font": {"color": "#333", "size": 14, "bold": True}
    }]
    st.session_state.vis_edges = []


def add_search_node(node_id, parent_ids, query=""):
    """添加搜索节点到图中"""
    # 截断过长的标签
    label = node_id[:12] + "..." if len(node_id) > 12 else node_id
    
    st.session_state.vis_nodes.append({
        "id": node_id,
        "label": f"🔍 {label}",
        "title": f"搜索: {query}",
        "color": {"background": "#06b6d4", "border": "#0891b2"},
        "shape": "box",
        "borderRadius": 8,
        "size": 25,
        "font": {"color": "#333", "size": 12}
    })
    
    for pid in parent_ids:
        st.session_state.vis_edges.append({"from": pid, "to": node_id})
    
    st.session_state.last_node_id = node_id


def update_node_summary(node_id, summary):
    """更新节点的 tooltip（摘要）"""
    for node in st.session_state.vis_nodes:
        if node['id'] == node_id:
            node['title'] = summary[:200] + "..." if len(summary) > 200 else summary
            break


def add_answer_node(parent_ids):
    """添加答案节点"""
    st.session_state.vis_nodes.append({
        "id": "answer",
        "label": "✨ Answer",
        "title": "最终答案",
        "color": {"background": "#f97316", "border": "#ea580c"},
        "shape": "circle",
        "size": 38,
        "font": {"color": "#333", "size": 14, "bold": True}
    })
    
    for pid in parent_ids:
        st.session_state.vis_edges.append({"from": pid, "to": "answer"})


def refresh_graph(placeholder, nodes=None, edges=None, height=500):
    """刷新 Memory Graph 可视化"""
    nodes = nodes if nodes is not None else st.session_state.vis_nodes
    edges = edges if edges is not None else st.session_state.vis_edges
    if nodes:
        html_content = render_graph_html(nodes, edges)
        with placeholder:
            components.html(html_content, height=height)


def auto_scroll_to_bottom():
    """注入 JS 自动滚动左列容器到底部"""
    components.html(
        '''<script>
        (function() {
            var container = parent.document.querySelector('[data-testid="stHorizontalBlock"] > [data-testid="column"]:first-child > div');
            if (container) {
                container.scrollTo({top: container.scrollHeight, behavior: "smooth"});
            }
        })();
        </script>''',
        height=0
    )


def parse_tool_call(content_text):
    """从 content 中解析 tool call"""
    try:
        if "<tool_call>" in content_text and "</tool_call>" in content_text:
            json_str = content_text.split("<tool_call>")[-1].split("</tool_call>")[0]
            tool_data = json5.loads(json_str)
            return tool_data.get('name'), tool_data.get('arguments', {})
    except Exception:
        pass
    return None, {}


def get_file_type_icon(file_type):
    """获取文件类型图标"""
    icons = {
        'image': '📷',
        'video': '🎬',
        'text': '📄'
    }
    return icons.get(file_type, '📎')


# ==================== 主页面渲染 ====================
def render_sidebar():
    """渲染侧边栏"""
    with st.sidebar:
        # 标题区
        st.markdown('<div class="sidebar-title">🧠 VimRAG</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-subtitle">Multimodal Memory Graph Agent</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        # 配置区
        st.markdown('<div class="config-header">🔧 Configuration</div>', unsafe_allow_html=True)
        
        base_url = st.text_input(
            "vLLM Base URL",
            value="https://dashscope.aliyuncs.com/compatible-mode/v1",
            key="base_url"
        )
        
        search_url = st.text_input(
            "Search Engine URL", 
            value="http://localhost:8001/search",
            key="search_url"
        )
        
        model_name = st.text_input(
            "Model Name",
            value="qwen3.5-plus",
            key="model_name"
        )
        
        api_key = st.text_input(
            "API Key",
            value=os.environ.get("DASHSCOPE_API_KEY", ""),
            type="password",
            key="api_key"
        )
        
        enable_thinking = st.toggle(
            "Enable Thinking",
            value=True,
            key="enable_thinking"
        )
        
        max_steps = st.slider(
            "Max Steps",
            min_value=5,
            max_value=30,
            value=20,
            key="max_steps"
        )
        
        search_top_k = st.slider(
            "Search Top-K",
            min_value=1,
            max_value=10,
            value=3,
            key="search_top_k"
        )
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        return {
            'base_url': base_url,
            'search_url': search_url,
            'model_name': model_name,
            'api_key': api_key,
            'enable_thinking': enable_thinking,
            'max_steps': max_steps,
            'search_top_k': search_top_k
        }


def render_thinking_block(thinking_text, container):
    """渲染思考块"""
    if thinking_text.strip():
        display_text = thinking_text.replace("<", "&lt;").replace(">", "&gt;")
        # 将连续多个换行符合并为最多一个换行
        display_text = re.sub(r'\n{3,}', '\n\n', display_text)
        # 去除首尾空白
        display_text = display_text.strip()
        html = f'''
        <details class="think-block" open>
            <summary class="think-label">💭 Thinking...</summary>
            <div class="think-content">{display_text}</div>
        </details>
        '''
        container.markdown(html, unsafe_allow_html=True)


def render_search_badge(node_id, query, container):
    """渲染搜索徽章"""
    safe_node_id = html_lib.escape(node_id or "")
    safe_query = html_lib.escape(query or "")
    html_str = f'''
    <div class="search-badge">
        <span class="icon">🧩</span>
        <span> 初始化记忆节点：</span>
        <span class="search-query">{safe_node_id}</span>
        <span> ｜ 🔎 检索：</span>
        <span class="search-query">{safe_query}</span>
    </div>
    '''
    container.markdown(html_str, unsafe_allow_html=True)


def render_file_tags(files, container):
    """渲染文件标签列表"""
    if not files:
        return
    
    tags_html = ""
    for fname, ftype in files:
        icon = get_file_type_icon(ftype)
        safe_fname = html_lib.escape(fname or "")
        tags_html += f'<span class="file-tag"><span class="type-icon">{icon}</span>{safe_fname}</span>'
    
    html_str = f'''
    <div class="files-container">
        <div class="files-label">📎 Retrieved Files:</div>
        {tags_html}
    </div>
    '''
    container.markdown(html_str, unsafe_allow_html=True)


def render_memory_card(summary, container):
    """渲染记忆卡片"""
    display_summary = html_lib.escape(summary or "")
    html_str = f'''
    <div class="memory-card">
        <div class="memory-header">
            <span>💾</span>
            <span>Memory Saved</span>
        </div>
        <div class="memory-content">{display_summary}</div>
    </div>
    '''
    container.markdown(html_str, unsafe_allow_html=True)


def render_answer_card(answer, container):
    """渲染答案卡片"""
    display_answer = answer.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
    html = f'''
    <div class="answer-card">
        <div class="answer-header">
            <span>✅</span>
            <span>Final Answer</span>
        </div>
        <div class="answer-content">{display_answer}</div>
    </div>
    '''
    container.markdown(html, unsafe_allow_html=True)


def render_error_card(error, container):
    """渲染错误卡片"""
    safe_error = html_lib.escape(str(error) if error else "")
    html_str = f'''
    <div class="error-card">
        ⚠️ Error: {safe_error}
    </div>
    '''
    container.markdown(html_str, unsafe_allow_html=True)


def build_graph_from_sample(graph_data):
    """从 answer 事件的 sample["graph"] 构建完整的图"""
    nodes = []
    edges = []
    
    for node in graph_data:
        node_id = node.get('id', '')
        parent_ids = node.get('parent_ids', [])
        
        if node_id == 'root':
            nodes.append({
                "id": "root",
                "label": "🌐 Query",
                "title": node.get('content', '用户问题'),
                "color": {"background": "#6366f1", "border": "#4f46e5"},
                "shape": "circle",
                "size": 38,
                "font": {"color": "#333", "size": 14}
            })
        elif node_id == 'answer':
            nodes.append({
                "id": "answer",
                "label": "✨ Answer",
                "title": "最终答案",
                "color": {"background": "#f97316", "border": "#ea580c"},
                "shape": "circle",
                "size": 38,
                "font": {"color": "#333", "size": 14}
            })
            for pid in parent_ids:
                edges.append({"from": pid, "to": "answer"})
        else:
            # 搜索节点
            label = node_id[:12] + "..." if len(node_id) > 12 else node_id
            summary = node.get('summary', node.get('query', ''))
            nodes.append({
                "id": node_id,
                "label": f"🔍 {label}",
                "title": summary[:200] + "..." if len(summary) > 200 else summary,
                "color": {"background": "#06b6d4", "border": "#0891b2"},
                "shape": "box",
                "borderRadius": 8,
                "size": 25,
                "font": {"color": "#333", "size": 12}
            })
            for pid in parent_ids:
                edges.append({"from": pid, "to": node_id})
    
    return nodes, edges


# ==================== 主函数 ====================
def main():
    # 注入 CSS
    inject_custom_css()
    
    # 初始化 session state
    init_session_state()
    
    # 渲染侧边栏并获取配置
    config = render_sidebar()
    
    # 顶部输入区（移到两列布局之前）
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown('<div class="input-label">💬 Ask VimRAG a question</div>', unsafe_allow_html=True)
    
    input_col1, input_col2 = st.columns([5, 1])
    
    with input_col1:
        question = st.text_input(
            "Question",
            value="阿里云有哪些AI产品？",
            placeholder="Type your question here...",
            key="question_input",
            label_visibility="collapsed"
        )
    
    with input_col2:
        submit = st.button("🚀 Submit", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 主内容区布局
    left_col, right_col = st.columns([55, 45], gap="large")
    
    # 左列 - Agent 推理流程
    with left_col:
        st.markdown('<div class="main-header"><span class="icon">🤖</span> Agent Reasoning Process</div>', unsafe_allow_html=True)
        
        # 推理过程显示区域
        process_container = st.container()
        
    # 右列 - Memory Graph
    with right_col:
        st.markdown('<div class="graph-header"><span>🕸️</span> Memory Graph</div>', unsafe_allow_html=True)
        graph_placeholder = st.empty()
        
        # 初始空状态
        if not st.session_state.vis_nodes:
            graph_placeholder.markdown('''
            <div class="empty-state">
                <div class="icon">🕸️</div>
                <div>Memory Graph will appear here<br>when you start a query</div>
            </div>
            ''', unsafe_allow_html=True)
    
    # 处理提交
    if submit and question.strip():
        # 重置状态
        reset_state()
        st.session_state.current_query = question
        st.session_state.is_running = True
        
        # 初始化 Agent
        try:
            agent = VimRAG(
                base_url=config['base_url'],
                search_url=config['search_url'],
                model_name=config['model_name'],
                api_key=config['api_key'],
                enable_thinking=config['enable_thinking'],
                max_mem_steps=config['max_steps'],
                search_top_k=config['search_top_k']
            )
        except Exception as e:
            with process_container:
                render_error_card(f"Failed to initialize agent: {str(e)}", st.empty())
            st.session_state.is_running = False
            return
        
        # 添加根节点
        add_root_node(question)
        
        # 更新图显示
        refresh_graph(graph_placeholder)
        
        # 流式处理事件
        thinking_text = ""
        content_text = ""
        current_search_query = ""
        
        # 占位符
        thinking_placeholder = None
        search_placeholder = None
        files_placeholder = None
        memory_placeholder = None
        answer_placeholder = None
        
        try:
            sample = {'query': question}
            
            for event in agent.run(sample):
                evt = event.get('event', '')
                
                if evt == 'think':
                    # 思考内容
                    thinking_text += event.get('content', '')
                    if thinking_placeholder is None:
                        with process_container:
                            thinking_placeholder = st.empty()
                    render_thinking_block(thinking_text, thinking_placeholder)
                    # 自动滚动到底部
                    auto_scroll_to_bottom()
                    
                elif evt == 'content':
                    # 模型输出（包含 tool call）
                    content_text += event.get('content', '')
                    
                    # 尝试解析 tool call 以实时更新图
                    tool_name, tool_args = parse_tool_call(content_text)
                    if tool_name == 'add_search_node' and tool_args:
                        node_id = tool_args.get('id', '')
                        parent_ids = tool_args.get('parent_ids', ['root'])
                        query = tool_args.get('query', '')
                        
                        # 检查节点是否已添加
                        existing_ids = [n['id'] for n in st.session_state.vis_nodes]
                        if node_id and node_id not in existing_ids:
                            add_search_node(node_id, parent_ids, query)
                            refresh_graph(graph_placeholder)
                    
                elif evt == 'search':
                    # 搜索开始
                    current_search_query = event.get('query', '')
                    current_node_id = event.get('node_id', '')
                    st.session_state.step_count += 1
                    
                    # 重置思考状态
                    thinking_text = ""
                    content_text = ""
                    thinking_placeholder = None  # 让下次 think 事件创建新占位符
                    
                    with process_container:
                        search_placeholder = st.empty()
                        render_search_badge(current_node_id, current_search_query, search_placeholder)
                        files_placeholder = st.empty()
                    # 自动滚动到底部
                    auto_scroll_to_bottom()
                    
                elif evt == 'search_done':
                    # 搜索完成，显示文件列表
                    results = event.get('results', {}).get('data', [])
                    file_list = []
                    for r in results:
                        fp = r.get('file_path', '')
                        fname = os.path.basename(fp) if fp else 'unknown'
                        ftype = r.get('type', 'text')
                        file_list.append((fname, ftype))
                    
                    if file_list and files_placeholder:
                        render_file_tags(file_list, files_placeholder)
                    # 自动滚动到底部
                    auto_scroll_to_bottom()
                    
                elif evt == 'memorize':
                    # 记忆摘要
                    summary = event.get('summary', '')
                    
                    with process_container:
                        memory_placeholder = st.empty()
                        render_memory_card(summary, memory_placeholder)
                    
                    # 更新最后一个节点的摘要
                    if st.session_state.last_node_id:
                        update_node_summary(st.session_state.last_node_id, summary)
                        refresh_graph(graph_placeholder)
                    
                    # 添加分割线
                    with process_container:
                        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                    
                    # 自动滚动到底部
                    auto_scroll_to_bottom()
                    
                    # 重置占位符
                    thinking_placeholder = None
                    search_placeholder = None
                    files_placeholder = None
                    memory_placeholder = None
                    
                elif evt == 'answer':
                    # 最终答案
                    answer_content = event.get('content', '')
                    sample_data = event.get('sample', {})
                    graph_data = sample_data.get('graph', [])
                    
                    # 从完整图数据重建（确保最终状态准确）
                    if graph_data:
                        nodes, edges = build_graph_from_sample(graph_data)
                        st.session_state.vis_nodes = nodes
                        st.session_state.vis_edges = edges
                        refresh_graph(graph_placeholder, nodes, edges)
                    
                    # 显示答案
                    with process_container:
                        answer_placeholder = st.empty()
                        render_answer_card(answer_content, answer_placeholder)
                    
                    # 自动滚动到底部
                    auto_scroll_to_bottom()
                    
                    st.session_state.is_running = False
                    
                elif evt == 'error':
                    # 错误
                    error_msg = event.get('content', 'Unknown error')
                    with process_container:
                        render_error_card(error_msg, st.empty())
                    
                elif evt == 'max_steps':
                    # 达到最大步数
                    with process_container:
                        render_error_card("Reached maximum reasoning steps", st.empty())
                    st.session_state.is_running = False
                    
        except Exception as e:
            with process_container:
                render_error_card(f"Agent error: {str(e)}", st.empty())
            st.session_state.is_running = False
    
    # 如果有已保存的图数据，显示它
    elif st.session_state.vis_nodes:
        refresh_graph(graph_placeholder)


if __name__ == "__main__":
    main()
