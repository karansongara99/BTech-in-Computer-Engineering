import os
import re
from typing import List, Dict
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

class MDIDMFowayChatbot:
    def __init__(self, txt_file_path: str = "mdidminfoway_placements_data.txt"):
        """
        Initialize the chatbot with text data and Gemini API
        """
        # Configure Gemini API
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Please set GEMINI_API_KEY in .env file")
        
        genai.configure(api_key=api_key)
        
        # Initialize Gemini model
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Load and preprocess text data
        self.txt_file_path = txt_file_path
        self.context_text = self.load_and_preprocess_text()
        
        # Chat history
        self.chat_history = []
        
        print("MDIDM Infoway Placements Chatbot Initialized!")
        print(f"Context loaded from: {txt_file_path}")
        print("Type 'exit' to quit the chatbot\n")
    
    def load_and_preprocess_text(self) -> str:
        """
        Load and preprocess the text file
        """
        try:
            with open(self.txt_file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            
            # Basic preprocessing
            # Remove excessive whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Ensure the text isn't too long (Gemini has context limits)
            # Keep it reasonable for the context window
            if len(text) > 100000:  # Approximate limit
                text = text[:100000] + "... [truncated for context limits]"
            
            return text
        
        except FileNotFoundError:
            print(f"Error: File {self.txt_file_path} not found.")
            print("Please run the web scraper first to generate the text file.")
            exit()
    
    def create_prompt(self, user_question: str) -> str:
        """
        Create a comprehensive prompt for Gemini
        """
        prompt = f"""You are a helpful assistant for MDIDM Infoway Placements information.
        
        CONTEXT FROM MDIDM INFOWAY PLACEMENTS PAGE:
        {self.context_text}
        
        USER QUESTION: {user_question}
        
        INSTRUCTIONS:
        1. Answer ONLY based on the provided context above
        2. If the information is not in the context, say "I don't have that information in my current data"
        3. Be specific and detailed in your responses
        4. Format your answer in a clear, readable way
        5. If mentioning companies, students, or placement statistics, be precise
        
        ANSWER:"""
        
        return prompt
    
    def ask_question(self, user_question: str) -> str:
        """
        Ask a question and get response from Gemini
        """
        # Create the prompt
        prompt = self.create_prompt(user_question)
        
        try:
            # Generate response
            response = self.model.generate_content(prompt)
            
            # Store conversation in history
            self.chat_history.append({
                'question': user_question,
                'answer': response.text
            })
            
            return response.text
        
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def chat_loop(self):
        """
        Start interactive chat loop
        """
        print("=" * 60)
        print("MDIDM INFOWAY PLACEMENTS CHATBOT")
        print("=" * 60)
        print("\nYou can ask questions about:")
        print("- Placement statistics")
        print("- Company partners")
        print("- Student achievements")
        print("- Placement procedures")
        print("- Training programs")
        print("- Any other information from the placements page")
        print("\n" + "-" * 60)
        
        while True:
            # Get user input
            user_input = input("\n📝 You: ").strip()
            
            # Check for exit command
            if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
                print("\n🤖 Chatbot: Thank you for using MDIDM Infoway Placements Chatbot!")
                break
            
            if not user_input:
                continue
            
            # Get response from chatbot
            print("\n🤖 Chatbot: ", end="")
            response = self.ask_question(user_input)
            print(response)
    
    def get_suggested_questions(self) -> List[str]:
        """
        Generate suggested questions based on the context
        """
        prompt = f"""Based on this context about placements, suggest 5 specific questions users might ask.
        
        Context: {self.context_text[:2000]}...
        
        Return only the questions as a numbered list."""
        
        try:
            response = self.model.generate_content(prompt)
            questions = response.text.strip().split('\n')
            return [q for q in questions if q.strip()]
        except:
            return [
                "1. What are the placement statistics for this year?",
                "2. Which companies recruit from MDIDM Infoway?",
                "3. What is the highest package offered?",
                "4. How does the placement process work?",
                "5. What training is provided for placements?"
            ]


# Alternative: Chatbot with conversation memory
class EnhancedMDIDMChatbot(MDIDMFowayChatbot):
    def __init__(self, txt_file_path: str = "mdidminfoway_placements_data.txt"):
        super().__init__(txt_file_path)
        # Initialize chat session for memory
        self.chat_session = self.model.start_chat(history=[])
    
    def create_prompt_with_history(self, user_question: str) -> str:
        """
        Create prompt with conversation history
        """
        # Build history context
        history_context = ""
        if self.chat_history:
            history_context = "\nPREVIOUS CONVERSATION:\n"
            for i, chat in enumerate(self.chat_history[-3:]):  # Last 3 exchanges
                history_context += f"Q{i+1}: {chat['question']}\n"
                history_context += f"A{i+1}: {chat['answer'][:200]}...\n"
        
        prompt = f"""You are a helpful assistant for MDIDM Infoway Placements information.
        
        CONTEXT FROM MDIDM INFOWAY PLACEMENTS PAGE:
        {self.context_text}
        
        {history_context}
        
        CURRENT USER QUESTION: {user_question}
        
        INSTRUCTIONS:
        1. Answer based on the provided context
        2. Consider the conversation history for context
        3. If information is not available, say so politely
        4. Be detailed and specific
        5. Maintain a helpful, professional tone
        
        ANSWER:"""
        
        return prompt
    
    def ask_question(self, user_question: str) -> str:
        """
        Ask question with conversation memory
        """
        prompt = self.create_prompt_with_history(user_question)
        
        try:
            # Send message to chat session
            response = self.chat_session.send_message(prompt)
            
            # Store in history
            self.chat_history.append({
                'question': user_question,
                'answer': response.text
            })
            
            return response.text
        
        except Exception as e:
            return f"Error: {str(e)}"


# Main execution
def main():
    """
    Main function to run the chatbot
    """
    print("Initializing MDIDM Infoway Placements Chatbot...\n")
    
    try:
        # Choose chatbot version
        print("Select chatbot version:")
        print("1. Basic Chatbot (Simple Q&A)")
        print("2. Enhanced Chatbot (With conversation memory)")
        
        choice = input("\nEnter choice (1 or 2): ").strip()
        
        if choice == "2":
            chatbot = EnhancedMDIDMChatbot()
        else:
            chatbot = MDIDMFowayChatbot()
        
        # Show suggested questions
        print("\n💡 Suggested Questions:")
        suggestions = chatbot.get_suggested_questions()
        for question in suggestions[:5]:
            print(f"  • {question}")
        
        # Start chat
        chatbot.chat_loop()
        
    except KeyboardInterrupt:
        print("\n\nChatbot session ended by user.")
    except Exception as e:
        print(f"\nError: {e}")
        print("Please ensure:")
        print("1. You have a valid Gemini API key in .env file")
        print("2. The text file exists in the same directory")
        print("3. You have an active internet connection")


# Batch question processing
def batch_process_questions():
    """
    Process multiple questions at once
    """
    chatbot = MDIDMFowayChatbot()
    
    questions = [
        "What is the placement percentage?",
        "Which companies visit for campus placements?",
        "What is the average salary package?",
        "How does MDIDM Infoway prepare students for placements?"
    ]
    
    print("Batch Processing Questions:\n")
    for i, question in enumerate(questions, 1):
        print(f"Q{i}: {question}")
        response = chatbot.ask_question(question)
        print(f"A{i}: {response[:200]}...\n")
        print("-" * 60)


if __name__ == "__main__":
    # Uncomment the function you want to run
    
    # Run interactive chatbot
    main()
    
    # Or run batch processing
    # batch_process_questions()