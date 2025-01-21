import streamlit as st
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import os
import tempfile
from math import ceil
import requests
import json

def call_whisper_api(audio_file_path, api_url, api_key=None, language=None):
    """
    Send audio file to custom Whisper API endpoint.
    
    Args:
        audio_file_path: Path to audio file
        api_url: Custom Whisper API endpoint URL
        api_key: Optional API key for authentication
        language: Optional language code
    Returns:
        str: Transcribed text
    """
    headers = {}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
    
    with open(audio_file_path, 'rb') as audio_file:
        files = {'audio': audio_file}
        data = {}
        if language:
            data['language'] = language
            
        try:
            response = requests.post(api_url, headers=headers, files=files, data=data)
            response.raise_for_status()
            return response.json().get('text', '')
        except requests.exceptions.RequestException as e:
            st.error(f"Error al llamar al API de Whisper: {str(e)}")
            return None

def split_audio(audio_segment, chunk_duration_ms=300000):  # 5 minutes chunks
    """
    Split audio into chunks of specified duration.
    """
    chunks = []
    for i in range(0, len(audio_segment), chunk_duration_ms):
        chunks.append(audio_segment[i:i + chunk_duration_ms])
    return chunks

def transcribe_chunk(audio_chunk, api_url, api_key=None, language=None):
    """
    Transcribe a single audio chunk using Whisper API.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        chunk_path = os.path.join(temp_dir, "temp_chunk.wav")
        audio_chunk.export(chunk_path, format="wav")
        return call_whisper_api(chunk_path, api_url, api_key, language)

def transcribe_video(video_file, api_url, api_key=None, language=None, progress_bar=None):
    """
    Transcribe audio from video file using custom Whisper API endpoint.
    """
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded video
            temp_video_path = os.path.join(temp_dir, "temp_video" + os.path.splitext(video_file.name)[1])
            with open(temp_video_path, "wb") as f:
                f.write(video_file.getbuffer())

            # Extract audio
            video = VideoFileClip(temp_video_path)
            if video.duration > 3600:
                st.warning(f"El video tiene una duración de {video.duration/3600:.1f} horas.")
            
            temp_audio_path = os.path.join(temp_dir, "temp_audio.wav")
            video.audio.write_audiofile(temp_audio_path, verbose=False, logger=None)
            
            # Convert and split audio
            sound = AudioSegment.from_file(temp_audio_path)
            chunks = split_audio(sound)
            
            # Process chunks
            transcribed_text = []
            for i, chunk in enumerate(chunks):
                if progress_bar is not None:
                    progress_bar.progress((i + 1) / len(chunks))
                    st.write(f"Procesando segmento {i+1} de {len(chunks)}...")
                
                chunk_text = transcribe_chunk(chunk, api_url, api_key, language)
                if chunk_text:
                    transcribed_text.append(chunk_text)
                    
                    # Update intermediate results
                    if i % 2 == 0:  # Update every 2 chunks
                        st.text_area(
                            "Transcripción en progreso...",
                            " ".join(transcribed_text),
                            height=150,
                            key=f"intermediate_{i}"
                        )
            
            return " ".join(transcribed_text)

    except Exception as e:
        st.error(f"Error durante la transcripción: {str(e)}")
        return None

def main():
    st.title("🎥 Transcriptor de Video con Whisper")
    st.write("Sube un video para transcribir su audio usando tu endpoint de Whisper")

    # API configuration
    with st.expander("Configuración del API de Whisper"):
        api_url = st.text_input(
            "URL del endpoint de Whisper",
            placeholder="https://tu-endpoint-whisper.com/v1/audio/transcriptions"
        )
        api_key = st.text_input(
            "API Key (opcional)",
            type="password",
            placeholder="Deja en blanco si no requiere autenticación"
        )
        if not api_key:
            api_key = None

    # Language selection
    languages = {
        "Español": "es",
        "English": "en",
        "Français": "fr",
        "Deutsch": "de",
        "Italiano": "it",
        "Português": "pt"
    }
    selected_language = st.selectbox(
        "Selecciona el idioma del video",
        options=list(languages.keys())
    )

    # File uploader
    supported_formats = ["mp4", "avi", "mov", "mkv", "wmv", "flv"]
    st.write(f"Formatos soportados: {', '.join(supported_formats)}")
    st.write("Límite de tamaño: 2GB")
    
    uploaded_file = st.file_uploader(
        "Sube tu archivo de video",
        type=supported_formats
    )

    if uploaded_file is not None:
        # Show video details
        st.write(f"Archivo subido: {uploaded_file.name}")
        file_size = uploaded_file.size / (1024 * 1024)
        st.write(f"Tamaño: {file_size:.2f} MB")
        
        if file_size > 1000:
            st.warning("Este es un archivo grande. El procesamiento puede tardar varios minutos.")

        # Verify API configuration
        if not api_url:
            st.error("Por favor, configura la URL del endpoint de Whisper")
        else:
            # Transcription button
            if st.button("Iniciar Transcripción"):
                progress_bar = st.progress(0)
                
                with st.spinner("Transcribiendo el video..."):
                    transcribed_text = transcribe_video(
                        uploaded_file,
                        api_url=api_url,
                        api_key=api_key,
                        language=languages[selected_language],
                        progress_bar=progress_bar
                    )
                    
                    if transcribed_text:
                        st.success("¡Transcripción completada!")
                        
                        # Show transcription
                        st.subheader("Texto transcrito:")
                        st.text_area(
                            "Resultado final",
                            transcribed_text,
                            height=300,
                            key="transcript"
                        )
                        
                        # Download button
                        st.download_button(
                            "Descargar transcripción",
                            transcribed_text,
                            file_name="transcripcion.txt",
                            mime="text/plain"
                        )

if __name__ == "__main__":
    main()
