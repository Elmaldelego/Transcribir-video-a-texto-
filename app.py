import os
# Establecer la variable de entorno al inicio
os.environ['KMP_DUPLICATE_LIB_OK']="TRUE"
import streamlit as st
from faster_whisper import WhisperModel
import os
import tempfile
from pydub import AudioSegment
import math
from datetime import timedelta
import subprocess

# Configurar la variable de entorno para OpenMP
os.environ['KMP_DUPLICATE_LIB_OK']="TRUE"

# Función para verificar si ffmpeg está instalado
def check_ffmpeg():
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True)
        return True
    except FileNotFoundError:
        return False

# Verificar ffmpeg al inicio
if not check_ffmpeg():
    st.error("FFmpeg no está instalado en el sistema. Por favor, contacta al administrador.")
    st.stop()
# Configuración de la página
st.set_page_config(
    page_title="Transcripción de Videos Largos",
    layout="wide",
)

def split_audio(audio_segment, chunk_duration=300000):  # 5 minutos en milisegundos
    """Divide el audio en segmentos más pequeños."""
    chunks = []
    total_duration = len(audio_segment)
    for i in range(0, total_duration, chunk_duration):
        chunk = audio_segment[i:i + chunk_duration]
        chunks.append(chunk)
    return chunks

def format_timestamp(seconds):
    """Convierte segundos a formato HH:MM:SS"""
    return str(timedelta(seconds=int(seconds)))

def extract_audio_from_video(video_path, progress_bar=None):
    """Extrae el audio de un archivo de video."""
    try:
        # Mostrar mensaje de procesamiento
        status_text = st.empty()
        status_text.text("Extrayendo audio del video...")
        
        # Cargar el video y extraer audio
        audio = AudioSegment.from_file(video_path)
        
        # Convertir a wav con una tasa de muestreo menor para reducir el tamaño
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_audio:
            status_text.text("Optimizando audio...")
            # Convertir a mono y reducir la tasa de muestreo
            audio = audio.set_channels(1).set_frame_rate(16000)
            audio.export(tmp_audio.name, format='wav')
            status_text.empty()
            return tmp_audio.name
    except Exception as e:
        st.error(f"Error al extraer audio: {str(e)}")
        return None

def process_audio_in_chunks(file_path, model, task="transcribe", language=None):
    """Procesa el audio en chunks y muestra el progreso."""
    audio = AudioSegment.from_file(file_path)
    chunks = split_audio(audio)
    
    # Crear barra de progreso
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    full_result = []
    for i, chunk in enumerate(chunks):
        # Actualizar progreso
        progress = (i + 1) / len(chunks)
        progress_bar.progress(progress)
        status_text.text(f"Procesando parte {i+1} de {len(chunks)}...")
        
        # Guardar chunk temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_chunk:
            chunk.export(tmp_chunk.name, format='wav')
            
            # Procesar chunk
            if task == "Transcripción":
                segments, _ = model.transcribe(tmp_chunk.name, language=language)
            else:
                segments, _ = model.translate(tmp_chunk.name)
            
            # Agregar resultados
            for segment in segments:
                full_result.append({
                    'start': segment.start + (i * 300),  # Ajustar tiempo según el chunk
                    'end': segment.end + (i * 300),
                    'text': segment.text
                })
            
            # Limpiar archivo temporal
            os.unlink(tmp_chunk.name)
    
    progress_bar.empty()
    status_text.empty()
    
    # Formatear resultado final
    return "\n".join([
        f"[{format_timestamp(segment['start'])} - {format_timestamp(segment['end'])}] {segment['text']}"
        for segment in full_result
    ])

def main():
    st.title("Transcripción y Traducción de Videos Largos")
    
    # Configuración
    task = st.selectbox(
        "Selecciona la tarea",
        ["Transcripción", "Traducción"]
    )
    
    model_size = st.selectbox(
        "Selecciona el modelo",
        ["large", "medium", "small", "tiny"]
    )
    
    language = st.selectbox(
        "Selecciona el idioma",
        ["Autodetect", "Spanish", "English"]
    )
    
    language = None if language == "Autodetect" else language
    
    # Subida de archivo
    st.info("Puedes subir videos de hasta 1 hora de duración.")
    uploaded_file = st.file_uploader(
        "Sube tu archivo de video", 
        type=['mp4', 'avi', 'mkv', 'mov']
    )
    
    if uploaded_file and st.button("Procesar"):
        with st.spinner('Iniciando procesamiento...'):
            # Guardar archivo temporal
            temp_path = None
            temp_audio_path = None
            
            try:
                # Guardar el video subido
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    temp_path = tmp_file.name
                
                # Extraer y procesar audio
                temp_audio_path = extract_audio_from_video(temp_path)
                if temp_audio_path:
                    # Cargar modelo
                    model = WhisperModel(model_size)
                    
                    # Procesar audio en chunks
                    result = process_audio_in_chunks(temp_audio_path, model, task, language)
                    
                    # Mostrar resultado
                    st.text_area("Resultado:", result, height=400)
                    
                    # Botón de descarga
                    st.download_button(
                        label="Descargar transcripción",
                        data=result,
                        file_name="transcripcion.txt",
                        mime="text/plain"
                    )
            
            finally:
                # Limpiar archivos temporales
                if temp_path and os.path.exists(temp_path):
                    os.unlink(temp_path)
                if temp_audio_path and os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)

if __name__ == "__main__":
    main()
