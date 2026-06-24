-- ═══════════════════════════════════════════════════════════════════════
-- supabase_setup.sql
-- ───────────────────
-- Run this ONCE in your Supabase project's SQL Editor before deploying.
-- Dashboard → SQL Editor → New query → paste this → Run.
-- ═══════════════════════════════════════════════════════════════════════

create table if not exists mnist_samples (
    id          bigint generated always as identity primary key,
    user_name   text        not null,
    label       smallint    not null check (label >= 0 and label <= 9),
    pixels      jsonb       not null,          -- 784 ints, row-major 28x28, 0-255
    confidence  real        default 0,
    created_at  timestamptz default now()
);

-- Index for fast per-digit / per-user queries (used by the dataset summary)
create index if not exists idx_mnist_samples_label on mnist_samples (label);
create index if not exists idx_mnist_samples_user   on mnist_samples (user_name);

-- ── Row Level Security ────────────────────────────────────────────────────
-- The app uses Supabase's "anon" public key, so RLS policies must
-- explicitly allow the operations the app needs: insert (save drawings)
-- and select (read dataset stats / exports). No update/delete from the
-- public app — that keeps the dataset append-only and safe from anyone
-- with the public anon key.

alter table mnist_samples enable row level security;

create policy "Allow public insert"
    on mnist_samples for insert
    to anon
    with check (true);

create policy "Allow public read"
    on mnist_samples for select
    to anon
    using (true);

-- That's it. Your table is ready — go to Project Settings → API to copy
-- your Project URL and anon public key into Streamlit secrets.
