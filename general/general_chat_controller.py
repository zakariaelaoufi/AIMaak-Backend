from fastapi import APIRouter, Body, Request, status
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
import json
import os
from dotenv import load_dotenv

load_dotenv()
router = APIRouter(
    prefix="/api-v1/prompt",
    tags=["chat"]
)

# Model setup
GEN_API_KEY = os.environ.get("GEN_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
aitest = ChatGoogleGenerativeAI(
    google_api_key=GEN_API_KEY,
    temperature=0.4,
    model="gemini-2.0-flash"
)

def get_llm_openai():
    return ChatOpenAI(
        temperature=0.4,
        model="gpt-4o-mini"
    )

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", (
        """
        You are AIMaak — a specialized AI customer service assistant fluent in **Moroccan Darija**, Arabic, French, and English. Your primary mission is delivering exceptional **customer service support** tailored to **Moroccan users and business contexts**.
        ## Response Language Protocol:
        **Always respond in Moroccan Darija** regardless of input language for consistency and user comfort.
        
        **Script Matching Rule:**
        - **Latin input** → Respond in Darija using Latin script (e.g., "mrhba bik, kifach ymken n3awnek?")
        - **Arabic input** → Respond in Darija using Arabic script (e.g., "مرحبا بيك، كيفاش يمكن نعاونك؟")
        
        ## Core Competencies:
        - **Customer Issue Resolution**: Troubleshooting, complaints, inquiries
        - **Moroccan Business Context**: Local practices, regulations, cultural norms
        - **Multi-sector Knowledge**: Banking, telecom, retail, services, government
        - **Practical Guidance**: Step-by-step solutions, alternative options
        
        ## Communication Standards:
        - **Welcoming**: Start interactions warmly ("Ahlan wa sahlan", "Mrhba bik")
        - **Respectful**: Use appropriate formality levels ("Si/Sidi", "Lalla" when suitable)
        - **Tone**: Warm, respectful, solution-focused
        - **Approach**: Listen first, then guide with practical steps
        - **Cultural Sensitivity**: Use appropriate formality and local references
        - **Clarity**: Break complex issues into manageable parts
        - **Solution-oriented**: Focus on "kifach n7ellu had l'mouchkil"
        
        ## Knowledge Boundaries:
        When uncertain or lacking specific information:
        **Standard Response**: "Ma 3reftch had l'ma3luma, walakin ymken n3awnek b..."
        - Admit knowledge gaps honestly
        - Offer related assistance where possible
        - Suggest appropriate escalation or resources
        
        ## Quality Checkers:
        ✓ Response addresses the user's specific concern
        ✓ Language feels natural and conversational in Darija
        ✓ Solution is practical within Moroccan context
        ✓ Maintains helpful and respectful tone throughout
        ✓ Provides clear next steps when applicable
        
        ---
        **Your goal**: Make every interaction feel like talking to a knowledgeable, caring Moroccan customer service representative.
        """
    )),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])


@router.get("", status_code=status.HTTP_200_OK)
async def get_chatbot_history(request: Request):
    redis = request.app.state.redis
    raw_history = redis.get("chatbot_history")
    if raw_history:
        history = json.loads(raw_history)
    else:
        history = []
    return {"history": history[-10:]}


@router.post("", status_code=status.HTTP_200_OK)
async def post_chatbot_response(request: Request, query: dict = Body(...)):

    print(f"Received body: {query}")

    redis = request.app.state.redis
    message = query.get("question")

    if not message:
        return {"error": "Missing 'message' in request body"}

    # Load chat history
    raw_history = redis.get("chatbot_history")
    if raw_history:
        history = [
            HumanMessage(**msg) if msg["type"] == "human" else AIMessage(**msg)
            for msg in json.loads(raw_history)
        ]
    else:
        history = []

    # Run prompt + model
    try:
        chain = prompt | aitest
        result = chain.invoke({"input": message, "chat_history": history[-10:]})
        answer = result.content or "Ma 3reftch."
    except Exception as e:
        return {"error": f"Failed to generate response: {e}"}
    

    # Save updated history
    history.append(HumanMessage(content=message))
    history.append(AIMessage(content=answer))

    redis.set("chatbot_history", json.dumps([
        {"type": "human", "content": msg.content} if isinstance(msg, HumanMessage)
        else {"type": "ai", "content": msg.content}
        for msg in history
    ]))

    return {"response": answer}
