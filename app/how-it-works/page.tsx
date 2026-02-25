import Image from "next/image";
import Link from "next/link";

export default function HowItWorksPage() {
  return (
    <main className="min-h-screen bg-zinc-50 text-zinc-900">
      <div className="mx-auto max-w-4xl px-6 py-10">
        <header className="mb-8 flex items-start justify-between gap-4">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">How it works</h1>
            <p className="mt-2 text-sm text-zinc-600">
              A technical (but friendly) explanation of how this app extracts dominant colours.
            </p>
          </div>

          <Link
            href="/"
            className="rounded-xl bg-white px-3 py-2 text-sm font-medium text-zinc-900 ring-1 ring-zinc-200 hover:bg-zinc-50"
          >
            Back
          </Link>
        </header>

        <section className="space-y-4 rounded-2xl bg-white p-6 shadow-sm ring-1 ring-zinc-200">
          <h2 className="text-lg font-semibold">Overview</h2>
          <p className="text-sm text-zinc-700 leading-6">
            This web app finds a small set of representative colours from your image using{" "}
            <span className="font-medium">K-Means clustering</span> in{" "}
            <span className="font-medium">HSV colour space</span>. In{" "}
            <span className="font-medium">Auto (Elbow)</span> mode, the number of colour clusters is
            chosen automatically using an elbow/knee detection method. Masks are generated only when
            you click <span className="font-medium">Show masks</span>.
          </p>

          <div className="rounded-xl bg-zinc-50 p-4 ring-1 ring-zinc-200">
            <h3 className="text-sm font-semibold">Privacy</h3>
            <p className="mt-2 text-sm text-zinc-700 leading-6">
              Your image is processed in memory to compute colours and (optionally) masks. The
              service is designed to not store uploaded images or build a gallery.
            </p>
          </div>
        </section>

        <section className="mt-8 space-y-4 rounded-2xl bg-white p-6 shadow-sm ring-1 ring-zinc-200">
          <h2 className="text-lg font-semibold">Step-by-step pipeline</h2>

          <ol className="list-decimal pl-5 space-y-3 text-sm text-zinc-700 leading-6">
            <li>
              <span className="font-medium">Resize</span>: The image is resized to a fixed width
              (e.g. 400px) to reduce computation while keeping aspect ratio.
            </li>
            <li>
              <span className="font-medium">Convert BGR → HSV</span>: HSV separates colour information
              (Hue) from intensity (Value), which often helps clustering.
            </li>
            <li>
              <span className="font-medium">Flatten pixels</span>: The image becomes a long list of
              pixels (each pixel is a 3D point: H, S, V).
            </li>
            <li>
              <span className="font-medium">Choose k</span>:
              <ul className="mt-2 list-disc pl-5 space-y-1">
                <li>
                  <span className="font-medium">Auto (Elbow)</span>: run K-Means with k=2..10, measure
                  SSE (inertia), then pick the “knee” point (KneeLocator).
                </li>
                <li>
                  <span className="font-medium">Manual</span>: you specify k (2–10).
                </li>
              </ul>
            </li>
            <li>
              <span className="font-medium">K-Means clustering</span>: assign each pixel to one of k
              clusters. Each cluster centre is treated as a dominant colour.
            </li>
            <li>
              <span className="font-medium">Ratios</span>: compute the fraction of pixels belonging to
              each cluster (percentage in the image).
            </li>
            <li>
              <span className="font-medium">Masks (optional)</span>: if you click{" "}
              <span className="font-medium">Show masks</span>, the app generates k partition images.
              Each partition keeps pixels of one cluster and paints other areas white. (To keep the
              response lighter, mask images are returned at a smaller width.)
            </li>
          </ol>
        </section>

        <section className="mt-8 space-y-4 rounded-2xl bg-white p-6 shadow-sm ring-1 ring-zinc-200">
          <h2 className="text-lg font-semibold">What is K-Means?</h2>
          <p className="text-sm text-zinc-700 leading-6">
            K-Means is an algorithm that groups points into k clusters by repeatedly:
            (1) assigning each point to the nearest cluster centre, and (2) updating each centre to
            the average of its assigned points. Here, each pixel is a point in HSV space.
          </p>

          <h2 className="text-lg font-semibold mt-6">What is the Elbow method?</h2>
          <p className="text-sm text-zinc-700 leading-6">
            As k increases, the clustering error (SSE / inertia) decreases. The elbow method chooses
            k at the point where additional clusters produce diminishing improvement. We detect that
            “knee” automatically (KneeLocator).
          </p>
        </section>

        <section className="mt-8 space-y-4 rounded-2xl bg-white p-6 shadow-sm ring-1 ring-zinc-200">
          <h2 className="text-lg font-semibold">Example flow (using Elbow method)</h2>

          {/* <p className="text-sm text-zinc-700 leading-6">
            You can add screenshots here to illustrate the user flow:
            input image → Auto (Elbow) results → Show masks results.
          </p> */}

          <div className="grid gap-4 md:grid-cols-3">
            <figure className="rounded-2xl bg-zinc-50 p-3 ring-1 ring-zinc-200">
              <div className="text-xs font-medium text-zinc-600 mb-2">A) Input image</div>
              <Image
                src="/docs/example-input.jpeg"
                alt="Example input"
                width={800}
                height={1000}
                className="h-auto w-full rounded-xl ring-1 ring-zinc-200 object-contain"
              />
              <figcaption className="mt-2 text-xs text-zinc-600">
                Upload an image in the main page.
              </figcaption>
            </figure>

            <figure className="rounded-2xl bg-zinc-50 p-3 ring-1 ring-zinc-200">
              <div className="text-xs font-medium text-zinc-600 mb-2">B) Auto (Elbow) result</div>
              <Image
                src="/docs/example-auto-result.jpeg"
                alt="Example auto result"
                width={800}
                height={1000}
                className="h-auto w-full rounded-xl ring-1 ring-zinc-200 object-contain"
              />
              <figcaption className="mt-2 text-xs text-zinc-600">
                The app chooses k and returns hex + ratios quickly.
              </figcaption>
            </figure>

            <figure className="rounded-2xl bg-zinc-50 p-3 ring-1 ring-zinc-200">
              <div className="text-xs font-medium text-zinc-600 mb-2">C) Show masks result</div>
              <Image
                src="/docs/example-masks.jpeg"
                alt="Example masks result"
                width={800}
                height={1000}
                className="h-auto w-full rounded-xl ring-1 ring-zinc-200 object-contain"
              />
              <figcaption className="mt-2 text-xs text-zinc-600">
                Clicking “Show masks” fetches partition masks using the fixed k.
              </figcaption>
            </figure>
          </div>

          <div className="rounded-xl bg-amber-50 p-4 ring-1 ring-amber-100">
            <p className="text-sm text-amber-900 leading-6">
              Note: Sometimes it takes a few minutes to process large images.
            </p>
          </div>
        </section>


        <footer className="mt-12 text-xs text-zinc-500">
          <p>
            copyrights &copy; 2026. built by <a href="https://nanothegigante.com">nano the gigante</a>.
            {/* Tip: If you publish globally, consider rate limiting and a clear privacy policy page. */}
          </p>
        </footer>
      </div>
    </main>
  );
}