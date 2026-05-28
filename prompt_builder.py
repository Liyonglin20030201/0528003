"""模块三：提示词装配模块

将用户问题与检索到的上下文组合成结构化提示词。
"""
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import Document


SYSTEM_TEMPLATE = """你是华智大学的智能校园问答助手，专门帮助学生解答关于奖学金、选课、培养管理等校园规章制度方面的问题。

请根据以下参考资料回答学生的问题。回答要求：
1. 准确引用规章制度中的具体条款
2. 语言简洁清晰，条理分明
3. 如果参考资料中没有相关信息，请诚实告知学生，并建议其咨询相关部门
4. 不要编造不存在的规定

参考资料：
{context}"""


def build_prompt() -> ChatPromptTemplate:
    """构建包含历史对话的聊天提示模板"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_TEMPLATE),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])
    return prompt


def format_context(docs: list[Document]) -> str:
    """将检索到的文档片段格式化为上下文文本"""
    context_parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "未知来源")
        source_name = source.split("\\")[-1].split("/")[-1]
        context_parts.append(f"[资料{i}] 来源：{source_name}\n{doc.page_content}")
    return "\n\n".join(context_parts)
