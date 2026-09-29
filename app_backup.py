import streamlit as st
import librosa
import numpy as np
import tempfile
import os
from gtts import gTTS
import speech_recognition as sr
from groq import Groq

# --- CONFIGURAÇÃO DA IA (GROQ) ---
CHAVE_GROQ = "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"
client = Groq(api_key=CHAVE_GROQ)

# --- DESCOBERTA AUTOMÁTICA BLINDADA E FILTRADA ---
@st.cache_data
def obter_lista_modelos():
    try:
        temp_client = Groq(api_key=CHAVE_GROQ)
        modelos = temp_client.models.list().data
        palavras_bloqueadas = ["whisper", "guard", "canopylabs", "vision", "embed", "llava"]
        return [m.id for m in modelos if not any(p in m.id.lower() for p in palavras_bloqueadas)]
    except Exception:
        return ["llama-3.1-8b-instant", "gemma2-9b-it", "mixtral-8x7b-32768"]

LISTA_MODELOS = obter_lista_modelos()

st.set_page_config(page_title="Treino de Pronúncia Dinâmico", page_icon="🗣️", layout="wide")
st.title("IdleChatter")
st.write("Fale sobre **qualquer assunto** em inglês. A IA vai conversar consigo com voz, avaliar o sotaque e manter o histórico!")

# --- MEMÓRIA DO CHAT AVANÇADA ---
if "mensagens" not in st.session_state:
    st.session_state.mensagens = []

if "ultimo_audio_id" not in st.session_state:
    st.session_state.ultimo_audio_id = None

if st.button("🔄 Iniciar Nova Conversa (Limpar Tudo)"):
    st.session_state.mensagens = []
    st.session_state.ultimo_audio_id = None
    st.rerun()

# --- RENDERIZA O HISTÓRICO DE CONVERSA (LINHA APÓS LINHA) ---
if st.session_state.mensagens:
    for msg in st.session_state.mensagens:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            
            # Se for a vez do utilizador, mostra as métricas e áudios
            if msg["role"] == "user" and "score" in msg:
                st.metric(label="Similaridade do Sotaque", value=f"{msg['score']:.1f}%")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Sua gravação:")
                    st.audio(msg["user_audio"], format="audio/wav")
                with col2:
                    st.write("Referência Nativa:")
                    st.audio(msg["native_audio"], format="audio/mp3")
                    
            # Se for a vez do tutor, mostra o áudio da resposta em inglês
            if msg["role"] == "assistant" and "audio" in msg:
                st.audio(msg["audio"], format="audio/mp3")
else:
    st.info("👋 O chat está vazio. Use o gravador no fundo da página e diga a sua primeira frase em inglês para iniciar!")


# --- TRUQUE DE INTERFACE: CONTENTOR RESERVADO ---
# Tudo o que for processado neste turno será desenhado AQUI (acima do microfone)
novo_turno_container = st.container()

# --- GRAVAÇÃO E INTERAÇÃO (SEMPRE NO FUNDO) ---
st.divider()

audio_file = st.audio_input("Grave a sua resposta para continuar a conversa:")

# --- FUNÇÕES DO MOTOR DE ÁUDIO ---
def comparar_audios(caminho_nativo, caminho_usuario):
    y1, sr1 = librosa.load(caminho_nativo, duration=10)
    y2, sr2 = librosa.load(caminho_usuario, duration=10)
    mfcc1 = librosa.feature.mfcc(y=y1, sr=sr1)
    mfcc2 = librosa.feature.mfcc(y=y2, sr=sr2)
    D, wp = librosa.sequence.dtw(X=mfcc1, Y=mfcc2, metric='cosine')
    distancia_media = D[-1, -1] / len(wp)
    score = max(0.0, min(100.0, 100 - (distancia_media * 200)))
    return score

def gerar_audio_nativo(texto):
    tts = gTTS(text=texto, lang='en', tld='us')
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tmp.close()
    tts.save(tmp.name)
    return tmp.name

def transcrever_audio(caminho_audio):
    r = sr.Recognizer()
    with sr.AudioFile(caminho_audio) as source:
        audio = r.record(source)
    try:
        return r.recognize_google(audio, language="en-US")
    except (sr.UnknownValueError, sr.RequestError):
        return None

# --- PROCESSAMENTO DO ÁUDIO NOVO ---
if audio_file is not None:
    id_audio_atual = f"{audio_file.name}_{audio_file.size}"
    
    if id_audio_atual != st.session_state.ultimo_audio_id:
        # Direciona todo o desenho da interface para o contentor acima do microfone
        with novo_turno_container:
            audio_bytes = audio_file.read()
            status_text = st.empty()
            path_user = None
            path_nativo = None
            path_audio_ia = None
            
            try:
                status_text.info("A ouvir a sua pronúncia...")
                tmp_user = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                tmp_user.write(audio_bytes)
                tmp_user.close()
                path_user = tmp_user.name
                
                texto_reconhecido = transcrever_audio(path_user)
                
                if texto_reconhecido:
                    st.session_state.ultimo_audio_id = id_audio_atual
                    
                    # 1. Processa e guarda a ação do utilizador
                    status_text.info("A comparar com o sotaque nativo...")
                    path_nativo = gerar_audio_nativo(texto_reconhecido)
                    score = comparar_audios(path_nativo, path_user)
                    with open(path_nativo, "rb") as f:
                        native_audio_bytes = f.read()
                    
                    st.session_state.mensagens.append({
                        "role": "user", 
                        "content": texto_reconhecido,
                        "score": score,
                        "user_audio": audio_bytes,
                        "native_audio": native_audio_bytes
                    })
                    
                    # Renderiza imediatamente a fala do utilizador na tela
                    with st.chat_message("user"):
                        st.write(texto_reconhecido)
                        st.metric(label="Similaridade do Sotaque", value=f"{score:.1f}%")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("Sua gravação:")
                            st.audio(audio_bytes, format="audio/wav")
                        with col2:
                            st.write("Referência Nativa:")
                            st.audio(native_audio_bytes, format="audio/mp3")
                            
                    # 2. Chama a IA para dar continuidade ao contexto
                    status_text.info("O tutor está a formular a resposta e a gerar voz...")
                    
                    historico_texto = ""
                    for msg in st.session_state.mensagens[:-1]:
                        prefixo = "Você" if msg["role"] == "user" else "Tutor"
                        historico_texto += f"{prefixo}: {msg['content']}\n"
                    
                    # PROMPT AGRESSIVO PARA OBRIGAR A TRADUÇÃO
                    prompt_ia = f"""
Você é um tutor de inglês focado em ajudar estudantes brasileiros.

Histórico da conversa:
{historico_texto}

O estudante acabou de dizer: "{texto_reconhecido}"

Responda rigorosamente em três parágrafos separados por quebras de linha (não use marcações complexas):
1. No primeiro parágrafo, escreva APENAS a sua resposta natural em INGLÊS, mantendo o diálogo vivo. (Não coloque títulos neste parágrafo para não atrapalhar o áudio).
2. No segundo parágrafo, escreva em PORTUGUÊS DO BRASIL uma avaliação pedagógica. Diga se a construção gramatical foi boa e sugira vocabulário.
3. No terceiro parágrafo, escreva a seção "Tradução da conversa:" e entregue a TRADUÇÃO REAL para o PORTUGUÊS DO BRASIL da frase do estudante e da sua resposta. É proibido repetir o texto em inglês aqui, entregue apenas as frases traduzidas para o português.
"""
                    texto_tutor = None
                    for modelo_teste in LISTA_MODELOS:
                        try:
                            resposta = client.chat.completions.create(
                                messages=[{"role": "user", "content": prompt_ia}],
                                model=modelo_teste
                            )
                            texto_tutor = resposta.choices[0].message.content
                            break
                        except Exception:
                            continue
                    
                    if texto_tutor:
                        # Extrai apenas o primeiro parágrafo (em inglês) para o gerador de voz
                        linhas_texto = [linha.strip() for linha in texto_tutor.split('\n') if linha.strip()]
                        fala_ingles = linhas_texto[0] if linhas_texto else "I'm sorry, I couldn't generate the audio."
                        
                        path_audio_ia = gerar_audio_nativo(fala_ingles)
                        with open(path_audio_ia, "rb") as f:
                            audio_ia_bytes = f.read()
                            
                        # Guarda e renderiza a resposta da IA com o áudio acoplado
                        st.session_state.mensagens.append({
                            "role": "assistant", 
                            "content": texto_tutor,
                            "audio": audio_ia_bytes
                        })
                        
                        with st.chat_message("assistant"):
                            st.write(texto_tutor)
                            st.audio(audio_ia_bytes, format="audio/mp3")
                    else:
                        st.error("Falha na conexão com a IA.")
                    
                    status_text.empty()
                    
                else:
                    status_text.empty()
                    st.error("Não conseguimos entender o áudio. Tente falar de forma mais clara.")
                    
            except Exception as e:
                status_text.empty()
                st.error(f"Ocorreu um erro: {e}")
                
            finally:
                try:
                    if path_user and os.path.exists(path_user): os.remove(path_user)
                    if path_nativo and os.path.exists(path_nativo): os.remove(path_nativo)
                    if path_audio_ia and os.path.exists(path_audio_ia): os.remove(path_audio_ia)
                except Exception:
                    pass