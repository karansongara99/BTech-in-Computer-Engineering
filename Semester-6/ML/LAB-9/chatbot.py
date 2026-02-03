import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load API key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

class SimplePlacementChatbot:
    def __init__(self):
        # Load text data
        with open("mdidminfoway_placements_data.txt", "r", encoding="utf-8") as f:
            self.context = f.read()[:50000]  # Limit context size
        
        # Initialize model
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
        print("Simple Placement Chatbot Ready!")
        print("Ask questions about MDIDM Infoway placements")
        print("Type 'quit' to exit\n")
    
    def answer_question(self, question):
        prompt = f"""Use this placement information to answer the question:
        
        {self.context}
        
        Question: {question}
        
        Answer based only on the above information. If not found, say "I don't have that information"."""
        
        response = self.model.generate_content(prompt)
        return response.text
    
    def chat(self):
        while True:
            question = input("You: ")
            if question.lower() in ['quit', 'exit', 'bye']:
                print("Goodbye!")
                break
            
            print("Bot: ", end="")
            answer = self.answer_question(question)
            print(answer)

# Quick start
if __name__ == "__main__":
    bot = SimplePlacementChatbot()
    bot.chat()