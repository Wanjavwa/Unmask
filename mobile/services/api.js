/**
 * Backend API for image verification.
 * Set API_BASE_URL to your machine's local IP when testing on a physical device.
 * Android emulator: use http://10.0.2.2:8000
 * iOS simulator: use http://localhost:8000
 */

import { Platform } from "react-native";

// For Android emulator: use 10.0.2.2
// For Android physical device: use your computer's LAN IP 
// For iOS simulator/web: use localhost
// Update ANDROID_PHYSICAL_IP if your computer's IP changes
const ANDROID_PHYSICAL_IP = "11.29.7.37"; // Your computer's IP on the network

const API_BASE_URL =
  __DEV__ && Platform.OS === "android"
    ? `http://${ANDROID_PHYSICAL_IP}:8000` // Use your computer's IP for physical device
    : __DEV__ && (Platform.OS === "ios" || Platform.OS === "web")
      ? "http://localhost:8000"
      : "http://YOUR_LOCAL_IP:8000";

/**
 * Upload image to backend and return detection result.
 * @param {string} uri - Local file URI (file:// or content://)
 * @returns {Promise<{ label: string, confidence: number, explanation: string, disclaimer?: string }>}
 * @throws {Error} When request fails or response is invalid
 */
export async function scanImage(uri) {
  const data = new FormData();

  if (Platform.OS === "web") {
    const response = await fetch(uri);
    const blob = await response.blob();
    const file = new File([blob], "upload.jpg", { type: "image/jpeg" });
    data.append("file", file);
  } else {
    data.append("file", {
      uri,
      name: "upload.jpg",
      type: "image/jpeg",
    });
  }

  const res = await fetch(`${API_BASE_URL}/detect-image`, {
    method: "POST",
    body: data,
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(
      res.status === 422 || res.status === 400
        ? "Invalid image. Please choose a valid photo."
        : res.status >= 500
          ? "Server error. Please try again later."
          : text || `Request failed (${res.status})`
    );
  }

  const json = await res.json();
  if (__DEV__) {
    console.log("Backend response:", json);
  }

  if (json == null || typeof json !== "object") {
    throw new Error("Invalid response from server.");
  }

  const label = json.label;
  const confidence = Number(json.confidence);
  const explanation = json.explanation;
  const disclaimer = json.disclaimer;

  if (typeof label !== "string" || !Number.isFinite(confidence)) {
    throw new Error("Invalid response format from server.");
  }

  return {
    label,
    confidence: Math.max(0, Math.min(1, confidence)),
    explanation: typeof explanation === "string" ? explanation : "",
    disclaimer: typeof disclaimer === "string" ? disclaimer : undefined,
  };
}
