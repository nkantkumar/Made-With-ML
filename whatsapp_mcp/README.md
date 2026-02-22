# WhatsApp MCP

Query your **WhatsApp export** (text, **photo**, **audio**, **video**) using **local Llama only** (Ollama). No cloud APIs.

- **Text**: Chat transcript is parsed and queried with Llama.
- **Photos**: Described with **Llama vision** (e.g. `llama3.2-vision`) and included in context for questions.
- **Audio / Video**: Listed by filename and type in the context (metadata only; no speech-to-text).

## Setup

1. **Export a chat from WhatsApp**  
   In the app: open chat → menu → Export chat → **Include media** (ZIP) or **Without media** (folder with `chat.txt`).

2. **Ollama + models**  
   ```bash
   ollama serve
   ollama run llama3.2
   ollama run llama3.2-vision   # for photo descriptions
   ```

3. **Env** (optional, in `.env`):  
   `OLLAMA_HOST`, `LLAMA_MODEL`, `LLAMA_VISION_MODEL=llama3.2-vision`

4. **Install**: `pip install ollama` (already in project `requirements.txt`).

## How to run and check

**CLI (from project root):**

```bash
# Point to a .zip, or to a folder that contains your export (e.g. store/)
# If you pass a folder (e.g. store), the first .zip inside it is used.
python -m whatsapp_mcp store "What did we decide about the meeting?"
python -m whatsapp_mcp store/whatsaap-chinease-teacher.zip "Summarize this chat"
python -m whatsapp_mcp path/to/chat_folder "List all photos mentioned in the chat"
```

**In code:**

```python
from whatsapp_mcp import load_export, query_chat, query_image, query_all

# Load export (ZIP or folder)
messages, media, chat_folder = load_export("path/to/WhatsApp_Chat.zip")
print(f"Messages: {len(messages)}, Media: {len(media)}")

# Query chat text only
answer = query_chat("Who sent the most messages?", messages)
print(answer)

# Ask about a single photo (Llama vision)
answer = query_image("path/to/Media/IMG-123.jpg", "What is in this image?")
print(answer)

# Query everything: chat + describe photos with vision + list audio/video
answer = query_all("path/to/export.zip", "Summarize the main topics and any photos.")
print(answer)
```

## Export format

- **ZIP**: Must contain a chat file (e.g. `_chat.txt`, `chat.txt`) and optionally a `Media/` folder.
- **Folder**: Same: a `.txt` chat file and optional `Media/` with images, audio, video.

Chat lines are expected in a form like:  
`[DD/MM/YYYY, HH:MM PM] Name: message` or `DD/MM/YYYY, HH:MM PM - Name: message`.

## API

- **`load_export(export_path)`** → `(messages, media_files, chat_folder)`
- **`query_chat(question, messages, config=None)`** → answer (text only)
- **`query_image(image_path, question="...", config=None)`** → answer (Llama vision)
- **`describe_photos(media, config=None, max_photos=50)`** → list of `(filename, description)`
- **`query_all(export_path, question, include_photo_descriptions=True, max_photos_to_describe=20)`** → one answer over chat + photo descriptions + audio/video list

Audio/video content is not transcribed (metadata only). For speech in audio/video you’d need a separate step (e.g. Whisper) and then add that text to the context.
