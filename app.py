from flask import Flask, request
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

app = Flask(__name__)

model = OllamaLLM(model='deepseek-r1:7b')

def generate_weather_prompt(temperature: float, humidity: float):
    """
    Generate a weather description prompt based on temperature and humidity.
    """
    prompt = f"""
    Describe the current weather conditions based on:
    - Temperature: {temperature}°C
    - Humidity: {humidity}%
    
    Provide a concise and natural weather report.
    """
    return prompt

@app.route("/weather_describe", methods=['GET'])
def weather_describe():
    temperature = request.args.get('temperature', type=float)
    humidity = request.args.get('humidity', type=float)

    if temperature is None or humidity is None:
        print("Error: Missing temperature or humidity parameters")
        return "Error: Missing parameters", 400

    prompt_text = generate_weather_prompt(temperature, humidity)
    prompt = ChatPromptTemplate.from_template(prompt_text)
    chain = prompt | model
    result = chain.invoke({})

    # Print result to terminal
    print("\n=== LLM Weather Description ===")
    print(result)
    print("================================\n")

    return "LLM response printed in terminal.", 200

if __name__ == "__main__":
    app.run(debug=True, port=8000, host='0.0.0.0')
