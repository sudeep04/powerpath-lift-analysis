"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

export function NavLink({
  href,
  children,
}: {
  href: string;
  children: React.ReactNode;
}) {
  const pathname = usePathname();
  const active = pathname === href;
  return (
    <Link
      href={href}
      aria-current={active ? "page" : undefined}
      className={`font-display text-sm font-semibold uppercase tracking-[0.16em] transition-colors ${
        active ? "text-accent" : "text-muted hover:text-ink"
      }`}
    >
      {children}
    </Link>
  );
}
