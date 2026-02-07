/**
 * Share-to-Verify: Used when the user shares an image from another app (Instagram, WhatsApp, etc.)
 * to Unmask on Android.
 *
 * - Picks the first image from the share intent
 * - On Android, converts content:// URIs to file:// in app cache so FormData can upload
 - Returns { uri, multipleImages, error } for the caller to update state
 */

import { Platform } from "react-native";

const IMAGE_MIME_PREFIX = "image/";

/**
 * Check if a share intent file is an image (JPEG, PNG, etc.)
 * @param { { mimeType?: string } } file
 * @returns {boolean}
 */
export function isImageFile(file) {
  const mime = (file?.mimeType || "").toLowerCase();
  return mime.startsWith(IMAGE_MIME_PREFIX);
}

/**
 * @param {string} uri
 * @param {string} [mimeType] - e.g. "image/jpeg"
 * @returns {Promise<string>} 
 */
export async function ensureFileUriForUpload(uri, mimeType = "image/jpeg") {
  if (!uri) return uri;
  if (Platform.OS !== "android") return uri;
  if (!uri.startsWith("content://")) return uri;

  try {
    const FileSystem = require("expo-file-system/legacy");
    const encoding =
      (FileSystem.EncodingType && FileSystem.EncodingType.Base64) || "base64";
    const ext = mimeType.includes("png") ? ".png" : ".jpg";
    const cachePath = `${FileSystem.cacheDirectory}unmask_shared_${Date.now()}${ext}`;
    const base64 = await FileSystem.readAsStringAsync(uri, { encoding });
    await FileSystem.writeAsStringAsync(cachePath, base64, { encoding });
    return cachePath.startsWith("file://") ? cachePath : `file://${cachePath}`;
  } catch (e) {
    if (__DEV__) console.warn("[shareHandler] ensureFileUriForUpload failed", e);
    return uri;
  }
}

/**
 * Get the first image from a share intent and a usable URI for upload
 * @param { { files?: Array<{ path?: string, filePath?: string, contentUri?: string, mimeType?: string }> } } shareIntent
 * @returns { Promise<{ uri: string | null, multipleImages: boolean, error: string | null }> }
 */
export async function getShareIntentImage(shareIntent) {
  const files = shareIntent?.files;
  if (!files?.length) {
    return { uri: null, multipleImages: false, error: "Unsupported share format." };
  }

  const imageFile = files.find(isImageFile) ?? files[0];
  if (!isImageFile(imageFile)) {
    return { uri: null, multipleImages: files.length > 1, error: "Unsupported share format." };
  }

  const rawUri =
    imageFile.path ??
    imageFile.filePath ??
    (Platform.OS === "android" ? imageFile.contentUri : null);
  if (!rawUri) {
    return { uri: null, multipleImages: files.length > 1, error: "Failed to load shared image." };
  }

  try {
    const uri = await ensureFileUriForUpload(rawUri, imageFile.mimeType || "image/jpeg");
    return {
      uri,
      multipleImages: files.length > 1,
      error: null,
    };
  } catch (e) {
    if (__DEV__) console.warn("[shareHandler] getShareIntentImage error", e);
    return {
      uri: null,
      multipleImages: files.length > 1,
      error: "Failed to load shared image.",
    };
  }
}
