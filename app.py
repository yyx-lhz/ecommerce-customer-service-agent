"""
Streamlit UI for the Cross-border Ecommerce Customer Service Agent.

Run with: streamlit run app.py
"""

import os
import sys
import time
import json
import traceback

import streamlit as st

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.graph import run_agent, run_agent_with_memory, clear_session, get_kb
from utils.mock_apis import lookup_order, track_shipment, check_stock


st.set_page_config(
    page_title="跨境电商智能客服 Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Generate a stable session ID for this browser tab
if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())[:8]

# ---------- Sidebar ----------

with st.sidebar:
    st.title("🤖 跨境电商客服 Agent")
    st.markdown("---")

    # API Key input
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=os.environ.get("OPENAI_API_KEY", ""),
        placeholder="sk-...",
        help="Your OpenAI API key is required for the agent to work.",
    )
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    st.markdown("---")

    # Quick test tools
    st.subheader("🔧 快速工具测试")

    test_order_id = st.text_input("订单号 (Order ID)", "ORD-1001")
    if st.button("查询订单", use_container_width=True):
        result = lookup_order(test_order_id)
        if result:
            st.success(f"✅ {result['product_name']} — {result['status']} — ${result['total']}")
            st.json(result)
        else:
            st.error("未找到订单")

    test_tracking = st.text_input("运单号 (Tracking No.)", "YT200")
    if st.button("追踪物流", use_container_width=True):
        # Find first matching tracking
        from utils.mock_apis import _ORDERS
        match = None
        for o in _ORDERS.values():
            if test_tracking in o.get("tracking_number", ""):
                match = o["tracking_number"]
                break
        if match:
            result = track_shipment(match)
            if result:
                st.success(f"📍 {result['status']} — {result['carrier']}")
                st.json(result["events"])
        else:
            st.info("未找到匹配的运单号")

    st.markdown("---")
    st.subheader("📦 产品目录")
    from utils.mock_apis import _products
    for pid, p in _products.items():
        stock_emoji = "🟢" if p["stock"] > 50 else "🟡" if p["stock"] > 0 else "🔴"
        with st.expander(f"{stock_emoji} {p['name_cn']} (${p['price']})"):
            st.write(p["description_cn"])
            st.write(f"库存: {p['stock']} | 运费: {'包邮' if p['shipping']['free_shipping'] else '$' + str(p['shipping'].get('shipping_cost', '?'))}")

    st.markdown("---")
    st.caption(f"Session: {st.session_state.session_id} | LangGraph + OpenAI + Chroma + Streamlit")

# ---------- Main chat area ----------

st.title("🛍️ 跨境电商智能客服 Agent")
st.caption("支持询盘咨询 · 订单查询 · 物流追踪 · 退换货政策 · 中英文双语")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": (
            "👋 你好！我是跨境电商智能客服助手。\n\n"
            "我可以帮你：\n"
            "- 📦 **查询产品信息** — 比如 \"蓝牙耳机多少钱？\"\n"
            "- 🔍 **追踪订单状态** — 比如 \"查一下 ORD-1001 的订单\"\n"
            "- 🚚 **追踪物流** — 提供运单号即可\n"
            "- ↩️ **退换货政策** — 了解退货流程\n\n"
            "请用中文或英文提问！"
        )}
    ]

# Display chat messages
for msg in st.session_state.messages:
    avatar = "🤖" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("请输入你的问题..."):
    if not api_key:
        st.error("⚠️ 请先在侧边栏输入 OpenAI API Key")
        st.stop()

    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Run agent
    with st.chat_message("assistant", avatar="🤖"):
        status_placeholder = st.empty()

        try:
            # Show progress
            status_placeholder.status("正在分析意图...", expanded=False)

            start_time = time.time()
            # Use multi-turn memory — pass session_id for conversation continuity
            result = run_agent_with_memory(prompt, st.session_state.session_id)
            elapsed = time.time() - start_time

            final_response = result.get("final_response", "")

            if not final_response:
                # Fallback: if graph didn't produce a response, show draft
                final_response = result.get("draft_response", "抱歉，我暂时无法处理这个请求，请稍后再试。")

            # Display response
            st.markdown(final_response)

            # Show debug info in expander
            with st.expander("🔍 调试信息", expanded=False):
                cols = st.columns(3)
                cols[0].metric("响应时间", f"{elapsed:.1f}s")
                cols[1].metric("意图", result.get("intent", "N/A"))
                cols[2].metric("自检通过", "✅" if result.get("reflection_passed") else "❌")

                if result.get("reflection_issues"):
                    st.warning("自检问题: " + "; ".join(result["reflection_issues"]))

                tabs = st.tabs(["工具调用", "RAG 上下文", "原始草稿"])
                with tabs[0]:
                    tool_results = result.get("tool_results", [])
                    if tool_results:
                        for tr in tool_results:
                            st.json(tr)
                    else:
                        st.info("无工具调用")
                with tabs[1]:
                    st.text(result.get("rag_context", "（空）")[:2000])
                with tabs[2]:
                    st.text(result.get("draft_response", "（空）"))

            st.session_state.messages.append({"role": "assistant", "content": final_response})

        except Exception as e:
            error_msg = f"❌ 发生错误: {str(e)}"
            st.error(error_msg)
            with st.expander("详细错误信息"):
                st.code(traceback.format_exc())
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

# ---------- Footer ----------
st.markdown("---")
st.caption("跨境电商智能客服 Agent · LangGraph + RAG + Streamlit · 多轮对话记忆 | 个人项目")
