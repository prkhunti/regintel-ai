import Link from "next/link";
import { useRouter } from "next/router";

const LINKS = [
  { href: "/documents", label: "Documents" },
  { href: "/eval", label: "Evaluation" },
  { href: "/audit", label: "Audit Log" },
];

export function Nav() {
  const { pathname } = useRouter();

  return (
    <header className="bg-white border-b border-gray-200 px-4 py-3 flex items-center gap-3">
      <Link href="/" className="flex items-center gap-2">
        <div className="w-7 h-7 rounded-lg bg-brand-600 flex items-center justify-center">
          <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h3.75M9 15h3.75M9 18h3.75m3 .75H18a2.25 2.25 0 002.25-2.25V6.108c0-1.135-.845-2.098-1.976-2.192a48.424 48.424 0 00-1.123-.08m-5.801 0c-.065.21-.1.433-.1.664 0 .414.336.75.75.75h4.5a.75.75 0 00.75-.75 2.25 2.25 0 00-.1-.664m-5.8 0A2.251 2.251 0 0113.5 2.25H15c1.012 0 1.867.668 2.15 1.586m-5.8 0c-.376.023-.75.05-1.124.08C9.095 4.01 8.25 4.973 8.25 6.108V8.25m0 0H4.875c-.621 0-1.125.504-1.125 1.125v11.25c0 .621.504 1.125 1.125 1.125h9.75c.621 0 1.125-.504 1.125-1.125V9.375c0-.621-.504-1.125-1.125-1.125H8.25zM6.75 12h.008v.008H6.75V12zm0 3h.008v.008H6.75V15zm0 3h.008v.008H6.75V18z" />
          </svg>
        </div>
        <span className="font-semibold text-gray-900 text-sm tracking-tight">RegIntel AI</span>
      </Link>

      <nav className="ml-auto flex items-center gap-4">
        {LINKS.map(({ href, label }) => (
          <Link
            key={href}
            href={href}
            className={`text-xs transition-colors ${
              pathname === href
                ? "text-brand-600 font-medium"
                : "text-gray-400 hover:text-brand-600"
            }`}
          >
            {label}
          </Link>
        ))}
      </nav>
    </header>
  );
}
