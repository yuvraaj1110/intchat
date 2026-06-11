# Deploying to Hugging Face Spaces (free, 16 GB RAM, no card)

This app is configured to run as a **Docker Space** on Hugging Face. The
`README.md` frontmatter and the `Dockerfile` are already set up. You just need
to create the Space, push the code, and add your Groq key.

## One-time setup

### 1. Create the Space
1. Go to https://huggingface.co/new-space
2. **Owner:** your account
3. **Space name:** `intchat` (or anything)
4. **License:** your choice
5. **SDK:** select **Docker** → **Blank** template
6. **Hardware:** `CPU basic` (free, 2 vCPU / 16 GB RAM)
7. **Visibility:** Public (free) or Private
8. Click **Create Space**

### 2. Add your Groq key as a secret
In the new Space: **Settings → Variables and secrets → New secret**
- **Name:** `GROQ_API_KEY`
- **Value:** your `gsk_...` key from https://console.groq.com/keys

(Secrets are injected as environment variables at runtime — `app/config.py`
reads `GROQ_API_KEY` automatically. Never commit the key.)

### 3. Push your code to the Space
HF Spaces is a git repo. Add it as a remote and push:

```bash
# Replace <user> and <space> with your values
git remote add space https://huggingface.co/spaces/<user>/<space>
git push space main
```

You'll be prompted for credentials:
- **Username:** your HF username
- **Password:** an HF **access token** (create one at
  https://huggingface.co/settings/tokens with "write" scope)

### 4. Watch it build
The Space's **Logs** tab shows the Docker build. It will:
1. Install dependencies
2. Run `build_kb --reset` (fetches sources, downloads the embedding model,
   builds Chroma) — takes a few minutes the first time
3. Start uvicorn on port 7860

When the build finishes, your app is live at
`https://huggingface.co/spaces/<user>/<space>` — share that URL with testers.

## Updating later

Any time you change code or `sources.yaml`:

```bash
git push space main
```

The Space rebuilds automatically.

## Notes

- **Sleeps after 48h** of inactivity, then wakes on the next visit (slow first
  request, then fast). Fine for a beta.
- **The build re-fetches live URLs** each time. If a government site is down
  during a build, that source is skipped (the build still succeeds).
- **Feedback log** (`feedback.jsonl`) lives on the Space's ephemeral disk and
  resets on rebuild. For durable feedback, download it from the Space or move to
  a database later (phase 2).
