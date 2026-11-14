# Deployment

The production setup is split:

- **Frontend:** Cloudflare Pages project `cutclipper` — https://cutclipper.pages.dev
- **Backend:** Render Docker web service `cutclipper-api` (Blueprint in `render.yaml`)

## Frontend

Build with the backend origin baked into the bundle:

```bash
cd frontend
VITE_API_BASE=https://cutclipper-api.onrender.com npm run build
npx wrangler pages deploy dist --project-name cutclipper --branch main
```

If the Render service gets a different hostname, rebuild with that URL. The
frontend calls it directly, so the backend must allow the Pages origin through
`CUTCLIPPER_CORS_ORIGINS` (the Blueprint currently allows all origins).

## Backend

Deploy the Blueprint from the repository:

https://render.com/deploy?repo=https://github.com/shahilkattel1-art/cutclipper

Render reads `render.yaml`, builds the root `Dockerfile`, and exposes:

- Health check: `GET /api/health`
- Public URL: `https://cutclipper-api.onrender.com` (unless the name is taken)

The Blueprint uses a `standard` instance with a 10 GB persistent disk at
`/data`. Whisper and FFmpeg need the memory; the free instance is too small.

## Cloudflare account

Wrangler is authenticated as `shahilkattel1@gmail.com` (account
`681001e5ea3a61747bcda7af22d6175d`). The account does not include Workers Paid,
so Cloudflare Containers cannot run the Python backend.

## Future frontend updates

Direct upload deployments are manual. To deploy every push automatically,
connect the Pages project to `shahilkattel1-art/cutclipper` in the Cloudflare
dashboard and set `VITE_API_BASE` to the Render URL in the build environment.
