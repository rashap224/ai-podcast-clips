import Link from "next/link";
import { auth } from "~/server/auth";
import { Button } from "~/components/ui/button";
import {
  Sparkles,
  Bot,
  Scissors,
  Video,
  AudioLines,
  ShieldCheck,
  Upload,
  Download,
  Brain,
  Zap,
  Target,
} from "lucide-react";

export default async function HomePage() {
  const session = await auth();
  const isLoggedIn = Boolean(session?.user);

  return (
    <main className="bg-white text-neutral-900">
      {/* Top Nav */}
      <header className="sticky top-0 z-20 border-b border-neutral-200 bg-white/80 backdrop-blur">
        <div className="container mx-auto flex h-16 items-center justify-between px-4">
          <Link href="/" className="flex items-center">
            <div className="text-xl font-medium tracking-tight">
              <span>podcast</span>
              <span className="px-1 text-neutral-400">/</span>
              <span className="font-light">clipper</span>
            </div>
          </Link>
          <nav className="hidden items-center gap-6 md:flex">
            <a href="#features" className="text-sm text-neutral-600 hover:text-neutral-900">Features</a>
            <a href="#how-it-works" className="text-sm text-neutral-600 hover:text-neutral-900">How it Works</a>
            <a href="#pricing" className="text-sm text-neutral-600 hover:text-neutral-900">Pricing</a>
          </nav>
          <div className="flex items-center gap-3">
            {isLoggedIn ? (
              <Button asChild variant="outline" className="hidden h-9 px-4 md:inline-flex">
                <Link href="/dashboard">Go to dashboard</Link>
              </Button>
            ) : (
              <>
                <Button asChild variant="ghost" className="h-9 px-3 text-sm">
                  <Link href="/login">Sign In</Link>
                </Button>
                <Button asChild className="h-9 px-4">
                  <Link href="/signup">Start Free Trial</Link>
                </Button>
              </>
            )}
          </div>
        </div>
      </header>

      {/* Hero */}
      <section className="relative border-b border-neutral-200">
        <div className="container mx-auto px-4 py-20">
          <div className="mx-auto max-w-4xl text-center">
            <span className="mx-auto inline-flex items-center gap-2 rounded-full border border-neutral-200 bg-white px-3 py-1 text-xs font-medium text-neutral-600">
              <Sparkles className="h-3.5 w-3.5" /> AI‑Powered Podcast Clipping
            </span>
            <h1 className="mt-6 text-4xl font-extrabold leading-[1.1] tracking-tight sm:text-6xl">
              Turn Your Podcasts Into Viral Short Clips
            </h1>
            <p className="mt-4 text-balance text-base text-neutral-600 sm:text-lg">
              Automatically detect engaging moments, keep speakers centered with face tracking, and export vertical clips for TikTok, Shorts, and Reels.
            </p>
            <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
              {isLoggedIn ? (
                <Button asChild className="h-11 px-6">
                  <Link href="/dashboard">Go to dashboard</Link>
                </Button>
              ) : (
                <>
                  <Button asChild className="h-11 px-6">
                    <Link href="/signup">Start Free Trial — 10 free credits</Link>
                  </Button>
                  <Button asChild variant="outline" className="h-11 px-6">
                    <a href="#pricing">View Pricing</a>
                  </Button>
                </>
              )}
            </div>
            <div className="mt-4 text-xs text-neutral-500">No watermarks. Keep full control of your content.</div>
          </div>
        </div>
      </section>

      {/* Features */}
      <section id="features" className="border-b border-neutral-200 bg-neutral-50/60">
        <div className="container mx-auto px-4 py-16">
          <div className="mx-auto mb-8 max-w-2xl text-center">
            <h2 className="text-3xl font-bold tracking-tight sm:text-4xl">Powerful AI Features</h2>
            <p className="mt-2 text-neutral-600">Everything you need to transform your long‑form podcasts into viral short‑form content.</p>
          </div>
          <div className="mx-auto grid max-w-6xl grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
            <FeatureCard icon={<Bot className="h-5 w-5" />} title="Viral Moment Detection" desc="LLM‑assisted ranking surfaces the most engaging stories, questions, and reactions." />
            <FeatureCard icon={<Scissors className="h-5 w-5" />} title="Auto Speaker Cropping" desc="Active speaker detection keeps the person talking perfectly framed." />
            <FeatureCard icon={<AudioLines className="h-5 w-5" />} title="Auto‑Generated Subtitles" desc="Whisper transcription produces accurate, punctuated captions." />
            <FeatureCard icon={<Video className="h-5 w-5" />} title="Vertical Format Optimization" desc="Export for 9:16 with safe margins for UI overlays across platforms." />
            <FeatureCard icon={<Zap className="h-5 w-5" />} title="GPU‑Accelerated Rendering" desc="Fast processing for longer episodes and batch jobs." />
            <FeatureCard icon={<Target className="h-5 w-5" />} title="Smart Content Analysis" desc="Filters filler moments and highlights parts that drive watch time." />
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section id="how-it-works" className="border-b border-neutral-200">
        <div className="container mx-auto px-4 py-16">
          <div className="mx-auto mb-10 max-w-2xl text-center">
            <h2 className="text-3xl font-bold tracking-tight sm:text-4xl">How It Works</h2>
            <p className="mt-2 text-neutral-600">Transform your podcasts into viral clips in just three simple steps.</p>
          </div>
          <div className="mx-auto grid max-w-5xl grid-cols-1 gap-4 md:grid-cols-3">
            <StepCard icon={<Upload className="h-6 w-6" />} title="Upload Your Podcast" desc="Drag and drop your MP4 (up to 500MB) or browse to select your episode." />
            <StepCard icon={<Brain className="h-6 w-6" />} title="AI Analysis & Processing" desc="We transcribe, detect viral moments, and identify the active speaker." />
            <StepCard icon={<Download className="h-6 w-6" />} title="Download Viral Clips" desc="Get multiple short‑form clips with subtitles and perfect framing." />
          </div>
        </div>
      </section>

      {/* Pricing */}
      <section id="pricing" className="border-b border-neutral-200 bg-neutral-50/60">
        <div className="container mx-auto px-4 py-16">
          <div className="mx-auto mb-10 max-w-2xl text-center">
            <h2 className="text-3xl font-bold tracking-tight sm:text-4xl">Simple, Transparent Pricing</h2>
            <p className="mt-2 text-neutral-600">Start with 10 free credits. Purchase more anytime — credits never expire.</p>
          </div>
          <div className="mx-auto grid max-w-6xl grid-cols-1 gap-6 md:grid-cols-3">
            <PriceCard title="Small Pack" price="$9.99" subtitle="50 credits" features={["No expiration", "Download all clips", "Email support"]} ctaText="Buy 50 Credits" />
            <PriceCard title="Medium Pack" price="$24.99" subtitle="150 credits" popular features={["No expiration", "Download all clips", "Priority support", "Advanced analytics"]} ctaText="Buy 150 Credits" />
            <PriceCard title="Large Pack" price="$69.99" subtitle="500 credits" features={["No expiration", "Download all clips", "Priority support", "Advanced analytics", "Custom branding"]} ctaText="Buy 500 Credits" />
          </div>
          <div className="mt-8 text-center text-xs text-neutral-500">Purchases are one‑time; no subscriptions. Manage billing inside your account.</div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-white">
        <div className="container mx-auto grid grid-cols-2 gap-8 px-4 py-12 md:grid-cols-4">
          <div>
            <div className="text-xl font-medium tracking-tight">
              <span>podcast</span>
              <span className="px-1 text-neutral-400">/</span>
              <span className="font-light">clipper</span>
            </div>
            <p className="mt-3 text-sm text-neutral-600">Transform your podcasts into viral clips with AI‑powered detection and automated editing.</p>
          </div>
          <div>
            <div className="text-sm font-semibold">Product</div>
            <ul className="mt-3 space-y-2 text-sm text-neutral-600">
              <li><a href="#features">Features</a></li>
              <li><a href="#pricing">Pricing</a></li>
              <li><Link href="/dashboard/billing">API</Link></li>
              <li><a href="#">Integrations</a></li>
            </ul>
          </div>
          <div>
            <div className="text-sm font-semibold">Support</div>
            <ul className="mt-3 space-y-2 text-sm text-neutral-600">
              <li><a href="#">Help Center</a></li>
              <li><a href="#">Contact Us</a></li>
              <li><a href="#">Status</a></li>
              <li><a href="#">Community</a></li>
            </ul>
          </div>
          <div>
            <div className="text-sm font-semibold">Company</div>
            <ul className="mt-3 space-y-2 text-sm text-neutral-600">
              <li><a href="#">About</a></li>
              <li><a href="#">Blog</a></li>
              <li><a href="#">Privacy</a></li>
              <li><a href="#">Terms</a></li>
            </ul>
          </div>
        </div>
        <div className="border-t border-neutral-200 py-6 text-center text-xs text-neutral-500">© {new Date().getFullYear()} podcast/clipper. All rights reserved.</div>
      </footer>
    </main>
  );
}

function FeatureCard({
  icon,
  title,
  desc,
}: {
  icon: React.ReactNode;
  title: string;
  desc: string;
}) {
  return (
    <div className="rounded-xl border border-neutral-200 bg-white p-4 shadow-sm">
      <div className="flex items-center gap-2 text-neutral-900">
        {icon}
        <h3 className="text-sm font-semibold">{title}</h3>
      </div>
      <p className="mt-2 text-sm text-neutral-600">{desc}</p>
    </div>
  );
}

function StepCard({
  icon,
  title,
  desc,
}: {
  icon: React.ReactNode;
  title: string;
  desc: string;
}) {
  return (
    <div className="rounded-xl border border-neutral-200 bg-white p-6 text-center shadow-sm">
      <div className="mx-auto mb-3 flex h-10 w-10 items-center justify-center rounded-full bg-neutral-100 text-neutral-700">
        {icon}
      </div>
      <h3 className="text-base font-semibold">{title}</h3>
      <p className="mt-2 text-sm text-neutral-600">{desc}</p>
    </div>
  );
}

function PriceCard({
  title,
  price,
  subtitle,
  features,
  ctaText,
  popular,
}: {
  title: string;
  price: string;
  subtitle: string;
  features: string[];
  ctaText: string;
  popular?: boolean;
}) {
  return (
    <div className={`relative flex flex-col rounded-xl border bg-white p-6 shadow-sm ${popular ? "border-2" : "border-neutral-200"}`}>
      {popular && (
        <div className="absolute left-1/2 top-0 -translate-y-1/2 -translate-x-1/2 rounded-full bg-neutral-900 px-3 py-1 text-xs font-medium text-white">Most Popular</div>
      )}
      <div className="flex-1">
        <div className="text-sm font-medium text-neutral-900">{title}</div>
        <div className="mt-2 text-4xl font-bold">{price}</div>
        <div className="text-sm text-neutral-500">{subtitle}</div>
        <ul className="mt-4 space-y-2 text-sm text-neutral-700">
          {features.map((f) => (
            <li key={f} className="flex items-center gap-2">
              <ShieldCheck className="h-4 w-4 text-neutral-800" />
              <span>{f}</span>
            </li>
          ))}
        </ul>
      </div>
      <div className="mt-6">
        <Button asChild className="w-full">
          <Link href="/dashboard/billing">{ctaText}</Link>
        </Button>
      </div>
    </div>
  );
}