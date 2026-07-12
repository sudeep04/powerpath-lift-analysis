import type { Metadata } from "next";
import { UploadForm } from "@/components/upload-form";

export const metadata: Metadata = {
  title: "Upload — PowerPath",
};

export default function UploadPage() {
  return <UploadForm />;
}
