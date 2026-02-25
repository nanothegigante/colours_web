"use client";

import { useMemo, useState } from "react";
import Link from "next/link";

type Mode = "auto" | "manual";
type ColourItem = { id: number; hex: string; ratio: number; mask_image?: string };
type ApiResponse = { k: number; colours: ColourItem[] };

function clampK(n: number) {
  return Math.min(10, Math.max(2, n));
}

export default function Page() {
  const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ?? "";

  const [file, setFile] = useState<File | null>(null);
  const [mode, setMode] = useState<Mode>("auto");
  const [k, setK] = useState<number>(5);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>("");
  const [data, setData] = useState<ApiResponse | null>(null);

  const previewUrl = useMemo(() => {
    if (!file) return "";
    return URL.createObjectURL(file);
  }, [file]);

  async function callExtract(file: File, mode: Mode, k: number, includeMasks: boolean) {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("mode", mode);
    if (mode === "manual") fd.append("k", String(k));
    fd.append("include_masks", includeMasks ? "true" : "false");

    const res = await fetch(`${API_BASE}/extract`, { method: "POST", body: fd });
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(`API error ${res.status}: ${text}`);
    }
    return (await res.json()) as ApiResponse;
  }

  async function onSubmit() {
    setError("");
    setData(null);
    if (!file) return setError("Please choose an image file first.");

    setLoading(true);
    try {
      // 初回は軽量
      const json = await callExtract(file, mode, clampK(k), false);
      setData(json);
    } catch (e: any) {
      setError(e?.message ?? "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

    // if (!API_BASE) {
    //   setError("API base URL is not set. Please configure NEXT_PUBLIC_API_BASE_URL.");
    //   return;
    // }
    // if (!file) {
    //   setError("Please choose an image file first.");
    //   return;
    // }

  async function loadMasks() {
    if (!file || !data) return;

    setError("");
    setLoading(true);
    try {
      // ★必要な時だけマスク取得（同じ画像を再POST）
      const json = await callExtract(file, mode, clampK(k), true);
      setData(json);
    } catch (e: any) {
      setError(e?.message ?? "Failed to load masks.");
    } finally {
      setLoading(false);
    }
  }


  function onReset() {
    setFile(null);
    setData(null);
    setError("");
  }

  return (
    <main className="min-h-screen bg-zinc-50 text-zinc-900">
      <div className="mx-auto max-w-4xl px-6 py-10">
        <header className="mb-8 flex items-start justify-between gap-4">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">Dominant Colours</h1>
            <p className="mt-2 text-sm text-zinc-600">
              Upload an image and extract dominant colours (hex, ratio, and partition masks).
            </p>
          </div>
          <Link
            href="/how-it-works"
            className="rounded-xl bg-white px-3 py-2 text-sm font-medium text-zinc-900 ring-1 ring-zinc-200 hover:bg-zinc-50"
          >
            How it works
          </Link>  
        </header>

        <section className="rounded-2xl bg-white shadow-sm ring-1 ring-zinc-200 p-6">
          <div className="grid gap-6 md:grid-cols-2">
            {/* Left: uploader + controls */}
            <div className="space-y-4">
              <div>
                <label className="text-sm font-medium">Image</label>
                <div className="mt-2">
                  <input
                    type="file"
                    accept="image/*"
                    onChange={(e) => setFile(e.target.files?.[0] ?? null)}
                    className="block w-full text-sm file:mr-4 file:rounded-xl file:border-0 file:bg-zinc-900 file:px-4 file:py-2 file:text-white hover:file:bg-zinc-800"
                  />
                  <p className="mt-2 text-xs text-zinc-500">
                    We process your image to extract colours. No gallery, no account.
                  </p>
                </div>
              </div>

              <div className="grid gap-4 sm:grid-cols-2">
                <div>
                  <label className="text-sm font-medium">Mode</label>
                  <div className="mt-2 flex gap-2">
                    <button
                      type="button"
                      onClick={() => setMode("auto")}
                      className={`rounded-xl px-3 py-2 text-sm ring-1 ${
                        mode === "auto"
                          ? "bg-zinc-900 text-white ring-zinc-900"
                          : "bg-white text-zinc-900 ring-zinc-200 hover:bg-zinc-50"
                      }`}
                    >
                      Auto (Elbow)
                    </button>
                    <button
                      type="button"
                      onClick={() => setMode("manual")}
                      className={`rounded-xl px-3 py-2 text-sm ring-1 ${
                        mode === "manual"
                          ? "bg-zinc-900 text-white ring-zinc-900"
                          : "bg-white text-zinc-900 ring-zinc-200 hover:bg-zinc-50"
                      }`}
                    >
                      Manual
                    </button>
                  </div>
                </div>

                <div className={mode === "manual" ? "" : "opacity-50"}>
                  <label className="text-sm font-medium">Clusters (k)</label>
                  <div className="mt-2 flex items-center gap-3">
                    <input
                      type="range"
                      min={2}
                      max={10}
                      value={k}
                      disabled={mode !== "manual"}
                      onChange={(e) => setK(parseInt(e.target.value, 10))}
                      className="w-full"
                    />
                    <span className="w-10 text-right text-sm tabular-nums">{clampK(k)}</span>
                  </div>
                  <p className="mt-1 text-xs text-zinc-500">2–10</p>
                </div>
              </div>

              <div className="flex gap-3">
                <button
                  type="button"
                  onClick={onSubmit}
                  disabled={loading}
                  className="rounded-xl bg-emerald-600 px-4 py-2 text-sm font-medium text-white hover:bg-emerald-700 disabled:opacity-50"
                >
                  {loading ? "Processing..." : "Extract"}
                </button>
                <button
                  type="button"
                  onClick={onReset}
                  disabled={loading}
                  className="rounded-xl bg-white px-4 py-2 text-sm font-medium text-zinc-900 ring-1 ring-zinc-200 hover:bg-zinc-50 disabled:opacity-50"
                >
                  Reset
                </button>
              </div>

              {error ? (
                <div className="rounded-xl bg-red-50 p-3 text-sm text-red-700 ring-1 ring-red-100">
                  {error}
                </div>
              ) : null}

              <div className="rounded-xl bg-zinc-50 p-3 text-xs text-zinc-600 ring-1 ring-zinc-200">
                <div className="font-medium text-zinc-800">API</div>
                <div className="mt-1 break-all">{API_BASE || "(not set)"}</div>
              </div>
            </div>

            {/* Right: preview */}
            <div className="space-y-3">
              <div className="text-sm font-medium">Preview</div>
              <div className="aspect-[4/5] w-full overflow-hidden rounded-2xl bg-zinc-100 ring-1 ring-zinc-200 flex items-center justify-center">
                {file ? (
                  // Use <img> to support local blob URL easily
                  <img src={previewUrl} alt="preview" className="h-full w-full object-contain" />
                ) : (
                  <div className="text-sm text-zinc-500">No image selected</div>
                )}
              </div>
              {data ? (
                <div className="text-xs text-zinc-600">
                  Returned k: <span className="font-medium text-zinc-900">{data.k}</span>
                </div>
              ) : null}
            </div>
          </div>
        </section>

        {/* Results */}
        {data ? (
          <section className="mt-8 space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold">Results</h2>
              <button
                type="button"
                onClick={loadMasks}
                disabled={loading || data.colours.every(c => c.mask_image)}
                className="rounded-xl bg-white px-4 py-2 text-sm font-medium text-zinc-900 ring-1 ring-zinc-200 hover:bg-zinc-50 disabled:opacity-50"
              >
                {data.colours.every(c => c.mask_image) ? "Masks loaded" : "Show masks"}
                
                {/* {loading ? "Loading masks..." : "Load masks"} */}
              </button>
            </div>

            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {data.colours.map((c, idx) => {
                const pct = (c.ratio * 100).toFixed(1);
                const maskSrc = c.mask_image ? `data:image/png;base64,${c.mask_image}` : "";
                
                return (
                  <div
                    key={idx}
                    className="rounded-2xl bg-white p-4 shadow-sm ring-1 ring-zinc-200"
                  >
                    <div className="flex items-center gap-3">
                      <div
                        className="h-10 w-10 rounded-xl ring-1 ring-zinc-200"
                        style={{ background: c.hex }}
                        // aria-label={`colour swatch ${c.hex}`}
                      />
                      <div className="min-w-0">
                        <div className="text-sm font-medium tabular-nums">{c.hex}</div>
                        <div className="text-xs text-zinc-600">{pct}%</div>
                      </div>
                    </div>

                    <div className="mt-3 overflow-hidden rounded-xl ring-1 ring-zinc-200 bg-zinc-50">
                      {c.mask_image ? (
                        <img src={maskSrc} alt={`mask ${idx}`} className="w-full h-auto" />
                      ) : (
                        <div className="p-3 text-xs text-zinc-500">
                          Mask not loaded. Click <span className="font-medium">"Show masks"</span>.
                        </div> 
                      )}
                      
                    </div>
                  </div>
                );
              })}
            </div>
          </section>
        ) : null}

        <footer className="mt-12 text-xs text-zinc-500">
          <p>
            copyrights &copy; 2026. built by <a href="https://nanothegigante.com">nano the gigante</a>
          </p>
        </footer>
      </div>
    </main>
  );
}
