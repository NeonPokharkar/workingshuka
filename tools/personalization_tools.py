from langchain.tools import ToolException, tool

@tool
def greetings_tool() -> str:
    """Provides a personalized greeting based on the time of day."""
    try:
        from datetime import datetime
        current_hour = datetime.now().hour
        if 5 <= current_hour < 12:
            return "Good morning! How can I assist you today?"
        elif 12 <= current_hour < 17:
            return "Good afternoon! What would you like to learn about today?"
        elif 17 <= current_hour < 22:
            return "Good evening! How can I help you unwind with some interesting facts?"
        else:
            return "Hello! It's quite late, but I'm here to help if you have any questions or just want to chat."
    except Exception as e:
        raise ToolException(f"Error generating greeting: {str(e)}")

tools_list = [greetings_tool]