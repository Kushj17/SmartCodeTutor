import os
import glob
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from crewai.tools import BaseTool
from crewai import Agent, Task, Crew, Process

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="AI Code Tutor", page_icon="💻", layout="wide")
st.title(" Multi-Agent Educational Code Explainer")
st.markdown("Powered by your fine-tuned CodeT5 and Gemini 3.1 Flash lite")

# Set your Gemini Key
os.environ["GEMINI_API_KEY"] = "YOUR_API_KEY"

# --- 2. THE CACHED MODEL LOADER (Crucial for your RAM) ---
@st.cache_resource
def load_local_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(current_dir, "..", "results", "codet5-beginner-explainer")
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint-*"))
    checkpoints.sort(key=lambda x: int(x.split('-')[-1]))
    latest_ckpt = checkpoints[-1] 

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(latest_ckpt)
    model = AutoModelForSeq2SeqLM.from_pretrained(latest_ckpt).to(device)
    
    return tokenizer, model, device

# Actually load the model when the app starts
tokenizer, model, device = load_local_model()

# --- 3. THE BULLETPROOF TOOL ---
class LocalCodeT5Tool(BaseTool):
    name: str = "Local_CodeT5_Explainer"
    description: str = "Pass raw python code to this tool to get a technical baseline explanation."

    def _run(self, code_snippet: str) -> str:
        prompt = f"explain python: {code_snippet}" 
        inputs = tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=128)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

codet5_explain_tool = LocalCodeT5Tool()

# --- 4. SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("⚙️ Personalization")
    user_level = st.selectbox(
        "Select User Level:", 
        ["Beginner", "Intermediate", "Advanced"]
    )
    st.markdown("---")
    st.markdown("### System Architecture")
    st.markdown("- **Local Grounding:** Fine-Tuned CodeT5")
    st.markdown("- **Reasoning:** Gemini")

# --- 5. MAIN UI & EXECUTION ---
code_input = st.text_area("Paste your Python code here:", height=200)

if st.button("Generate Explanation", type="primary"):
    if not code_input.strip():
        st.warning("Please enter some code first!")
    else:
        with st.spinner(f"Agents are analyzing for a {user_level}..."):
            
            # Define Agents
            
            # Explainer Agent: The technical lead
            explainer_agent = Agent(
                role='Technical Code Analyst',
                goal='Generate a technically accurate baseline explanation of the provided code.',
                backstory='You specialize in understanding raw code logic. You use local specialized models to extract meaning.',
                tools=[codet5_explain_tool],
                llm="gemini/gemini-3.1-flash-lite",
                verbose=True
            )

            # Simplifier Agent: The educational expert (Phase 6 implementation)
            simplifier_agent = Agent(
                role='Educational Personalization Expert',
                goal=f'Adapt technical explanations for a {user_level} programmer.',
                backstory=f'You are a CS professor who knows how to teach {user_level}s. You use analogies and simple terms.',
                llm="gemini/gemini-3.1-flash-lite",
                verbose=True
            )

            # Complexity Agent: The DSA specialist
            complexity_agent = Agent(
                role='Algorithm Specialist',
                goal='Provide precise Big O Time and Space complexity analysis.',
                backstory='You live and breathe Data Structures and Algorithms. You provide concise complexity metrics.',
                llm="gemini/gemini-3.1-flash-lite",
                verbose=True
            )

            # Critic Agent: The Final Reviewer
            critic_agent = Agent(
                role='Content Quality Lead',
                goal='Ensure the final explanation is cohesive, accurate, and perfectly formatted.',
                backstory='You ensure the output follows the required structure: Summary, Step-by-Step, Example, and Complexity.',
                llm="gemini/gemini-3.1-flash-lite",
                verbose=True
            )

            # Define Tasks
            task1 = Task(
                description=f"Analyze this code using the Local_CodeT5_Explainer tool: {code_input}",
                agent=explainer_agent,
                expected_output="A technical summary of the code logic."
            )

            task2 = Task(
                description=f"Take the technical summary and simplify it for a {user_level} user. Use analogies if helpful.",
                agent=simplifier_agent,
                expected_output=f"A {user_level}-friendly, step-by-step guide."
            )

            task3 = Task(
                description=f"Determine the Time and Space complexity for: {code_input}",
                agent=complexity_agent,
                expected_output="Time: O(n), Space: O(n) for recursive stack."
            )

            task4 = Task(
                description="Combine all previous outputs into a structured Markdown report.",
                agent=critic_agent,
                expected_output="The final finalized personalized code explanation report.",
                context=[task1, task2, task3]
            )

            # Phase 5: Multi-Agent System Assembly
            crew = Crew(
                agents=[explainer_agent, simplifier_agent, complexity_agent, critic_agent],
                tasks=[task1, task2, task3, task4],
                process=Process.sequential
            )
            result = crew.kickoff()
            
            # Display Results
            st.success("Analysis Complete!")
            st.markdown("### 📝 Your Personalized Explanation")
            st.markdown(result.raw)