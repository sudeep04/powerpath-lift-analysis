import type { Metadata } from "next";
import { Player } from "@/components/player";

export const metadata: Metadata = {
  title: "PowerPath — Analysis",
};

export default async function VideoPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  return <Player videoId={id} />;
}
