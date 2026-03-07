# Final Deployment Step-by-Step

Amazing! The codebase has been fully prepared. I’ve added everything needed `.gitignore` (so we safely ignore API keys), generated `requirements.txt`, and added a `Procfile` for Render.

Please follow these exact steps to complete the deployment:

---

## Step 1: Push the Code to GitHub

1. Go to [GitHub - Create a New Repository](https://github.com/new).
2. Name it `accessify-live` (make it Public or Private, your choice) and click **Create repository**.
3. Open a new Terminal on your computer specifically in your `accessify-live` root folder.
4. Run these exact commands (copy and paste them one by one):
   ```bash
   git init
   git branch -M main
   git add .
   git commit -m "Ready for production"
   git remote add origin https://github.com/YOUR_GITHUB_USERNAME/accessify-live.git
   git push -u origin main
   ```
   *(Make sure to replace the `YOUR_GITHUB_USERNAME` URL with the actual URL GitHub gives you)*

---

## Step 2: Deploy the Backend to Render (Free)

1. Go to [Render.com](https://render.com/) and sign up with GitHub.
2. Click **New +** at the top right, and select **Web Service**.
3. Connect your new `accessify-live` GitHub repository.
4. Fill in the deployment details:
   * **Name:** `accessify-backend`
   * **Language:** Python
   * **Region:** (Pick whichever is closest to you)
   * **Branch:** `main`
   * **Root Directory:** `backend` *(CRITICAL: Put `backend` here)*
   * **Build Command:** `pip install -r requirements.txt`
   * **Start Command:** `uvicorn app:app --host 0.0.0.0 --port $PORT`
   * **Instance Type:** Free
5. Scroll down and click **Advanced**.
6. Click **Add Environment Variable**. Add **two** variables:
   * Key: `FIREBASE_STORAGE_BUCKET` / Value: *(Get this from your frontend `.env.local`)*
   * Key: `FIREBASE_SERVICE_ACCOUNT_JSON` / Value: *(Open your `serviceAccountKey.json` on your PC, copy ALL the text inside, and paste it here.)*
7. Click **Create Web Service**. (It will take about 5 minutes to build).

*Once done, Render will give you a live URL like `https://accessify-backend.onrender.com`. Copy this!*

---

## Step 3: Deploy the Frontend to Vercel (Free)

1. Go to [Vercel.com](https://vercel.com/) and sign up with GitHub.
2. Click **Add New... > Project**.
3. Import your `accessify-live` GitHub repository.
4. In the "Configure Project" screen:
   * **Framework Preset:** Vite
   * **Root Directory:** Edit this and select `frontend`.
5. Open the **Environment Variables** section. Add all the `VITE_FIREBASE_...` keys from your local `.env.local` file one by one.
6. Add one **more** incredibly important variable:
   * Key: `VITE_API_URL`
   * Value: *(Paste the URL Render gave you in Step 2, e.g., `https://accessify-backend.onrender.com`)*
   * Key: `VITE_WS_URL`
   * Value: *(Paste the explicit Render WebSocket URL, e.g., `wss://accessify-backend.onrender.com`)*
7. Click **Deploy**.

---

You're done! Now you can visit your Vercel frontend URL, and the real-time classrooms will be talking to your cloud backend. Let me know if you run into any errors or need me to check the logs of your services!
