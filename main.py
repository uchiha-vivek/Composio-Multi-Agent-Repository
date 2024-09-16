# main.py

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import aiofiles
import os
from composio_autogen import ComposioToolSet, App
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager, ConversableAgent
from textwrap import dedent
import dotenv

dotenv.load_dotenv()

app = FastAPI()

config_list = [
    {
        "model": "llama3-70b-8192",
        "api_key": os.environ.get("GROQ_API_KEY"),
        "api_type": "groq",
    }
]

llm_config = {"config_list": config_list, "timeout": 300, "temperature": 0}

data_analysis_agent = ConversableAgent(
    "Data Insights Specialist",
    llm_config=llm_config,
    description=dedent("""\
        Provides Summary Report of key metrics and trends gotten 
        from data provided by the File Parser agent.
    """),
    system_message=dedent("""\
        You are a data insights specialist who analyzes data from the File Parser agent and provides a detailed summary of key metrics and trends.
        Return the summary report as context to the user agent.
    """),
    max_consecutive_auto_reply=2,
)

da_proxy_agent = ConversableAgent(
    "File Parser and Extractor",
    llm_config=llm_config,
    description=dedent("""\
        Reads, Parses CSV files and Returns a formatted output using
        the column names in the first row of the CSV file.
    """),
    system_message=dedent("""\
        You are a file parser that reads CSV files and returns a formatted output based on the columns in the first row.
        Stop the process with the message "TERMINATE".
    """),
    human_input_mode="NEVER",
    code_execution_config={"use_docker": False},
    max_consecutive_auto_reply=5,
)

user_agent = UserProxyAgent(
    "User",
    description=dedent("""\
        Collects and returns the final response from the Data Insights Specialist.
    """),
    human_input_mode="NEVER",
    code_execution_config={"use_docker": False},
    max_consecutive_auto_reply=5,
)

group_chat = GroupChat(
    agents=[data_analysis_agent, da_proxy_agent, user_agent],
    messages=[],
    send_introductions=True,
    max_round=5,
)

group_chat_manager = GroupChatManager(
    groupchat=group_chat,
    llm_config=llm_config,
)

composio_toolset = ComposioToolSet(api_key=os.getenv("COMPOSIO_API_KEY"))

composio_toolset.register_tools(
    apps=[App.FILETOOL],
    caller=data_analysis_agent,
    executor=da_proxy_agent,
)

@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...)):
    file_location = f"csv_files/{file.filename}"
    
    # Save the file to a directory
    async with aiofiles.open(file_location, 'wb') as out_file:
        content = await file.read()
        await out_file.write(content)
    
    # Generate the summary
    task = f"Generate a Detailed Summary report on the csv file {file_location} in my current directory "
    
    response = user_agent.initiate_chat(
       recipient=group_chat_manager,
       message=task,
       summary_method="last_msg"
    )
    summary_report = response.summary

    # Document writing process
    doc_writer_agent = AssistantAgent(  
           "Document Writer",  
           description=dedent("""  
                               Starts the write operation to a Google Document using the available tools provided.  
                               """),  
           llm_config=llm_config,  
           human_input_mode="NEVER",  
           system_message=dedent("""  
                                   You are a document writer agent that writes a summary report to a Google Document using the available tools provided.  
                                   """),  
           max_consecutive_auto_reply=2  
       ) 

    content_writer = UserProxyAgent(
        "Content Writer",
        description="Writes content to a Google Document",
        human_input_mode="NEVER",
        code_execution_config={"use_docker": False},
        max_consecutive_auto_reply=5
        )

    user_agent = UserProxyAgent(
        "User",
        description=dedent("""\
                            Ensures the summary report is written to a Google Document.
                            """),
        human_input_mode="NEVER",
        code_execution_config={"use_docker": False},
        max_consecutive_auto_reply=5
        )
    group_chat = GroupChat(
        agents=[doc_writer_agent, content_writer, user_agent],
        messages=[],
        send_introductions=True,
        max_round=5
    )

    group_chat_manager = GroupChatManager(
        groupchat=group_chat,
        llm_config=llm_config,
    )
    composio_toolset.register_tools(
        apps=[App.GOOGLEDOCS],
        caller=doc_writer_agent,
        executor=content_writer,
    )
    
    res2 = user_agent.initiate_chat(
        recipient=group_chat_manager,
        message=f"Write the summary report '{summary_report}' to a Google Document named 'User Feedback Report'.",
    )

    return JSONResponse(content={"filename": file.filename, "summary": summary_report, "doc_status": res2})
