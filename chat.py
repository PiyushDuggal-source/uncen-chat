import streamlit as st
import ollama
from melo.api import TTS

speed = 1.2

# CPU is sufficient for real-time inference.
# You can set it manually to 'cpu' or 'cuda' or 'cuda:0' or 'mps'
device = "auto"  # Will automatically use GPU if available

# English


# American accent
def synthesize(text, speed):
    model = TTS(language="EN", device=device)
    speaker_ids = model.hps.data.spk2id
    output_path = f"audio{st.session_state.count}.wav"
    model.tts_to_file(
        text, speaker_ids["EN-US"], output_path, speed=speed, format="wav"
    )

    return output_path


st.title("Echo Bot")

options = [obj["name"] for obj in ollama.list()["models"]]

# Add the selectbox to the sidebar
model_name = st.sidebar.selectbox("Select a model", options)
system_prompt = st.sidebar.text_input("Enter the system prompt (Your resume details):")


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "count" not in st.session_state:
    st.session_state.count = 0

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            st.audio(
                synthesize(message["content"], speed),
                format="audio/wav",
                start_time=0,
                sample_rate=None,
                end_time=None,
                loop=False,
            )


# React to user input
if prompt := st.chat_input("Write something to UncenChat:"):

    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    if len(system_prompt.strip()) > 0:
        st.session_state.messages.append({"role": "system", "content": system_prompt})

    output_path = f"audio{st.session_state.count}.wav"
    # Display assistant response in chat message container
    with st.chat_message("assistant"):

        stream = ollama.generate(
            model=str(model_name),
            prompt=prompt,
            system=system_prompt,
            stream=True,
            keep_alive=0,
            # images=[bytes_data] if bytes_data is not None else None,
        )

        text = ""

        text = st.write_stream(map(lambda x: x["response"], stream))
        synthesize(text, speed)

    audio_file = f"audio{st.session_state.count}.wav"
    st.audio(
        audio_file,
        format="audio/wav",
        start_time=0,
        sample_rate=None,
        end_time=None,
        loop=False,
        autoplay=True,
    )
    st.session_state.count += 1

    # Add assistant response to chat history
    st.session_state.messages.append(
        {"role": "assistant", "content": text, output_path: output_path}
    )
