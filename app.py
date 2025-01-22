import os
import streamlit as st
from faster_whisper import WhisperModel
import tempfile
from pydub import AudioSegment
import warnings

# Ignorar advertencias específicas de pydub
warnings.filterwarnings('ignore', category=SyntaxWarning)

# Configurar la variable de entorno para OpenMP
os.environ['KMP_DUPLICATE_LIB_OK']="TRUE"

def format_time(seconds):
    """Convierte segundos a formato HH:MM:SS"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def extract_audio(video_path):
    """Extrae el audio de un archivo de video."""
    try:
        with st.spinner('Extrayendo audio del video...'):
            audio = AudioSegment.from_file(video_path)
            # Convertir a mono y ajustar la tasa de muestreo
            audio = audio.set_channels(1).set_frame_rate(16000)
            
            # Guardar en archivo temporal
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
                audio.export(temp_audio.name, format='wav')
                return temp_audio.name
    except Exception as e:
        st.error(f"Error al procesar el audio: {str(e)}")
        return None

def transcribe_audio(audio_path, model, language=None):
    """Transcribe el audio usando Whisper."""
    try:
        segments, _ = model.transcribe(audio_path, language=language)
        return [(segment.start, segment.end, segment.text) for segment in segments]
    except Exception as e:
        st.error(f"Error en la transcripción: {str(e)}")
        return []

def main():
    st.title("Transcriptor de Video a Texto")
    
    # Configuraciones
    model_size = st.selectbox(
        "Selecciona el tamaño del modelo",
        ["tiny", "base", "small", "medium", "large"]
    )
    
    language = st.selectbox(
        "Selecciona el idioma",
        ["Autodetectar", "Español", "English"]
    )
    
    # Mapeo de idiomas
    language_map = {
        "Autodetectar": None,
        "Español": "es",
        "English": "en"
    }
    
    # Subida de archivo
    video_file = st.file_uploader("Sube tu archivo de video", type=['mp4', 'avi', 'mov', 'mkv'])
    
    if video_file and st.button("Transcribir"):
        try:
            # Guardar el archivo de video
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
                temp_video.write(video_file.read())
                video_path = temp_video.name
            
            # Extraer audio
            audio_path = extract_audio(video_path)
            if not audio_path:
                st.error("No se pudo extraer el audio del video.")
                return
            
            # Cargar modelo
            with st.spinner('Cargando modelo de transcripción...'):
                model = WhisperModel(model_size)
            
            # Transcribir
            with st.spinner('Transcribiendo...'):
                results = transcribe_audio(audio_path, model, language_map[language])
            
            if results:
                # Mostrar resultados
                transcript = "\n".join([
                    f"[{format_time(start)} - {format_time(end)}] {text}"
                    for start, end, text in results
                ])
                
                st.text_area("Transcripción:", transcript, height=400)
                
                # Botón de descarga
                st.download_button(
                    label="Descargar transcripción",
                    data=transcript,
                    file_name="transcripcion.txt",
                    mime="text/plain"
                )
            
            # Limpiar archivos temporales
            try:
                os.unlink(video_path)
                os.unlink(audio_path)
            except:
                pass
                
        except Exception as e:
            st.error(f"Error durante el proceso: {str(e)}")

if __name__ == "__main__":
    main()
