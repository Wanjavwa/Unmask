import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  AccessibilityInfo,
  Animated,
  BackHandler,
  Dimensions,
  Image,
  Modal,
  Platform,
  Pressable,
  StatusBar,
  StyleSheet,
  Text,
  View,
} from "react-native";
import { SafeAreaProvider, SafeAreaView } from "react-native-safe-area-context";
import * as ImagePicker from "expo-image-picker";
import { useShareIntent } from "expo-share-intent";
import { scanImage } from "./services/api";
import { getShareIntentImage } from "./src/utils/shareHandler";
import { LinearGradient } from "expo-linear-gradient";
import {
  useFonts,
  Orbitron_600SemiBold,
  Orbitron_700Bold,
} from "@expo-google-fonts/orbitron";
import {
  useFonts as useExo2,
  Exo2_500Medium,
  Exo2_600SemiBold,
} from "@expo-google-fonts/exo-2";
import {
  useFonts as useInter,
  Inter_400Regular,
  Inter_500Medium,
} from "@expo-google-fonts/inter";

// ─── Design system (premium futuristic) ─────────────────────────────────────
const COLORS = {
  bgMain: "#0f1629",
  bgSoft: "#151b32",
  bgDeep: "#0d1225",
  card: "#ffffff",
  cardShadow: "rgba(10,28,255,0.12)",
  accentBlue: "#0a1cff",
  accentBlueDark: "#0612b8",
  accentBlueSoft: "rgba(10,28,255,0.15)",
  accentCyan: "#00f0ff",
  accentCyanDark: "#00b7c5",
  glowBlue: "rgba(10,28,255,0.6)",
  glowCyan: "rgba(0,240,255,0.5)",
  textPrimary: "#0b0d1a",
  textOnDark: "#ffffff",
  textOnDarkSecondary: "#b8c4e0",
  textOnDarkMuted: "#8a9bc4",
  textSecondary: "#3d4556",
  textMuted: "#64748b",
  realGlow: "rgba(34,197,94,0.5)",
  aiGlow: "rgba(239,68,68,0.5)",
};

// ─── Layout constants (responsive + consistent) ──────────────────────────────
const SPACING = 16;
const RADIUS = 22;
const BUTTON_HEIGHT = 56;
const MAX_CONTENT_WIDTH = 420;
const CONTENT_WIDTH_PERCENT = "92%";

async function ensureGalleryPermission() {
  const { status, canAskAgain } = await ImagePicker.getMediaLibraryPermissionsAsync();
  if (status === "granted") return { ok: true };

  const req = await ImagePicker.requestMediaLibraryPermissionsAsync();
  if (req.status === "granted") return { ok: true };

  return {
    ok: false,
    message: canAskAgain
      ? "Gallery access is required to select images."
      : "Gallery access is required to select images. Please enable it in Settings.",
  };
}

const { width: SCREEN_WIDTH } = Dimensions.get("window");
const PARTICLE_COUNT = 20;

export default function App() {
  const [selectedUri, setSelectedUri] = useState(null);
  const [message, setMessage] = useState("");
  const [aboutVisible, setAboutVisible] = useState(false);
  const [verification, setVerification] = useState(null);
  const [scanError, setScanError] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [launchedFromShare, setLaunchedFromShare] = useState(false);
  const [shareBannerMessage, setShareBannerMessage] = useState(null);
  const [shareError, setShareError] = useState(null);
  const [reduceMotion, setReduceMotion] = useState(false);
  const consumedShareRef = useRef(false);

  const { hasShareIntent, shareIntent, resetShareIntent } = useShareIntent({
    resetOnBackground: false,
  });

  const [fontsLoadedOrbitron] = useFonts({ Orbitron_600SemiBold, Orbitron_700Bold });
  const [fontsLoadedExo2] = useExo2({ Exo2_500Medium, Exo2_600SemiBold });
  const [fontsLoadedInter] = useInter({ Inter_400Regular, Inter_500Medium });
  const fontsLoaded = fontsLoadedOrbitron && fontsLoadedExo2 && fontsLoadedInter;

  const pressAnim = useRef(new Animated.Value(1)).current;
  const glowPulse = useRef(new Animated.Value(0)).current;
  const previewFade = useRef(new Animated.Value(0)).current;
  const resultShimmer = useRef(new Animated.Value(0)).current;
  const analyzingPulse = useRef(new Animated.Value(0)).current;
  const scrollY = useRef(new Animated.Value(0)).current;

  const heroTitleScale = useRef(new Animated.Value(0.7)).current;
  const heroTitleOpacity = useRef(new Animated.Value(0)).current;
  const heroTitleTranslateY = useRef(new Animated.Value(50)).current;
  const heroTitleRotateX = useRef(new Animated.Value(-12)).current;
  const heroGlowFlicker = useRef(new Animated.Value(0)).current;
  const heroTaglineOpacity = useRef(new Animated.Value(0)).current;
  const heroPanelOpacity = useRef(new Animated.Value(0)).current;
  const heroShimmerOffset = useRef(new Animated.Value(0)).current;

  const imageTranslateY = useRef(new Animated.Value(36)).current;
  const imageBob = useRef(new Animated.Value(0)).current;
  const resultBadgePulse = useRef(new Animated.Value(1)).current;
  const confidenceBarScale = useRef(new Animated.Value(0)).current;
  const buttonGlowPress = useRef(new Animated.Value(0)).current;
  const analyzingShimmer = useRef(new Animated.Value(0)).current;

  const particleAnims = useRef(
    Array.from({ length: PARTICLE_COUNT }, () => new Animated.Value(0))
  ).current;

  useEffect(() => {
    AccessibilityInfo.getReduceMotionEnabled?.()
      .then(setReduceMotion)
      .catch(() => {});
    const sub = AccessibilityInfo.addEventListener?.("reduceMotionChanged", setReduceMotion);
    return () => sub?.remove?.();
  }, []);

  useEffect(() => {
    if (
      !hasShareIntent ||
      !shareIntent?.files?.length ||
      consumedShareRef.current
    ) {
      return;
    }
    let cancelled = false;
    consumedShareRef.current = true;
    setShareError(null);
    setShareBannerMessage(null);

    getShareIntentImage(shareIntent)
      .then(({ uri, multipleImages, error }) => {
        if (cancelled) return;
        resetShareIntent(true);
        if (error) {
          setShareError(error);
          return;
        }
        if (uri) {
          setSelectedUri(uri);
          setLaunchedFromShare(true);
          setShareBannerMessage(
            multipleImages
              ? "Multiple images received, analyzing first image only."
              : "Image received from share menu."
          );
          setMessage("Image received. Analyzing…");
        } else {
          setShareError("Failed to load shared image.");
        }
      })
      .catch(() => {
        if (!cancelled) {
          setShareError("Failed to load shared image.");
          resetShareIntent(true);
        }
      });

    return () => { cancelled = true; };
  }, [hasShareIntent, shareIntent, resetShareIntent]);

  const duration = (d) => (reduceMotion ? Math.min(d, 150) : d);

  const buttonShadow = useMemo(() => {
    const shadowOpacity = Animated.multiply(
      glowPulse.interpolate({ inputRange: [0, 1], outputRange: [0.4, 0.7] }),
      buttonGlowPress.interpolate({ inputRange: [0, 1], outputRange: [1, 1.4] })
    );
    const shadowRadius = glowPulse.interpolate({
      inputRange: [0, 1],
      outputRange: [14, 26],
    });
    return { shadowOpacity, shadowRadius };
  }, [glowPulse, buttonGlowPress]);

  useEffect(() => {
    if (!fontsLoaded) return;
    Animated.parallel([
      Animated.timing(heroTitleScale, {
        toValue: 1,
        duration: duration(1100),
        useNativeDriver: true,
      }),
      Animated.timing(heroTitleOpacity, {
        toValue: 1,
        duration: duration(700),
        useNativeDriver: true,
      }),
      Animated.timing(heroTitleTranslateY, {
        toValue: 0,
        duration: duration(1100),
        useNativeDriver: true,
      }),
      Animated.timing(heroTitleRotateX, {
        toValue: 0,
        duration: duration(1000),
        useNativeDriver: true,
      }),
      Animated.timing(heroPanelOpacity, {
        toValue: 1,
        duration: duration(600),
        useNativeDriver: true,
      }),
    ]).start();
    Animated.sequence([
      Animated.delay(duration(200)),
      Animated.timing(heroGlowFlicker, {
        toValue: 1,
        duration: duration(120),
        useNativeDriver: true,
      }),
      Animated.timing(heroGlowFlicker, {
        toValue: 0.3,
        duration: duration(100),
        useNativeDriver: true,
      }),
      Animated.timing(heroGlowFlicker, {
        toValue: 1,
        duration: duration(150),
        useNativeDriver: true,
      }),
    ]).start();
    Animated.sequence([
      Animated.delay(duration(450)),
      Animated.timing(heroTaglineOpacity, {
        toValue: 1,
        duration: duration(550),
        useNativeDriver: true,
      }),
    ]).start();
    Animated.timing(heroShimmerOffset, {
      toValue: 1,
      duration: duration(900),
      useNativeDriver: true,
    }).start();
  }, [fontsLoaded, reduceMotion]);

  useEffect(() => {
    if (reduceMotion) return;
    const loop = Animated.loop(
      Animated.sequence([
        Animated.timing(glowPulse, {
          toValue: 1,
          duration: 1800,
          useNativeDriver: false,
        }),
        Animated.timing(glowPulse, {
          toValue: 0.45,
          duration: 1800,
          useNativeDriver: false,
        }),
      ])
    );
    loop.start();
    return () => loop.stop();
  }, [glowPulse, reduceMotion]);

  useEffect(() => {
    if (reduceMotion) return;
    particleAnims.forEach((anim, i) => {
      const delay = i * 180;
      const dur = 8000 + i * 600;
      Animated.loop(
        Animated.sequence([
          Animated.delay(delay),
          Animated.timing(anim, {
            toValue: 1,
            duration: dur,
            useNativeDriver: true,
          }),
          Animated.timing(anim, { toValue: 0, duration: 0, useNativeDriver: true }),
        ])
      ).start();
    });
  }, [reduceMotion]);

  useEffect(() => {
    if (!selectedUri) {
      previewFade.setValue(0);
      imageTranslateY.setValue(36);
      imageBob.setValue(0);
      setVerification(null);
      setScanError(null);
      setIsAnalyzing(false);
      confidenceBarScale.setValue(0);
      return;
    }
    previewFade.setValue(0);
    imageTranslateY.setValue(36);
    imageBob.setValue(0);
    setVerification(null);
    setScanError(null);
    setIsAnalyzing(true);
    confidenceBarScale.setValue(0);
    Animated.parallel([
      Animated.timing(previewFade, {
        toValue: 1,
        duration: duration(420),
        useNativeDriver: true,
      }),
      Animated.timing(imageTranslateY, {
        toValue: 0,
        duration: duration(500),
        useNativeDriver: true,
      }),
    ]).start();
    const bobbing = reduceMotion
      ? null
      : Animated.loop(
          Animated.sequence([
            Animated.timing(imageBob, {
              toValue: 1,
              duration: 2200,
              useNativeDriver: true,
            }),
            Animated.timing(imageBob, {
              toValue: 0,
              duration: 2200,
              useNativeDriver: true,
            }),
          ])
        );
    bobbing?.start();

    scanImage(selectedUri)
      .then((res) => {
        setIsAnalyzing(false);
        setVerification({
          label: res.label,
          confidence: res.confidence,
          explanation: res.explanation || "",
          disclaimer: res.disclaimer,
        });
        resultShimmer.setValue(0);
        resultBadgePulse.setValue(1);
        Animated.parallel([
          Animated.timing(resultShimmer, {
            toValue: 1,
            duration: duration(400),
            useNativeDriver: true,
          }),
          Animated.spring(resultBadgePulse, {
            toValue: 1.12,
            useNativeDriver: true,
            speed: 18,
            bounciness: 4,
          }),
        ]).start();
        setTimeout(() => {
          Animated.spring(resultBadgePulse, {
            toValue: 1,
            useNativeDriver: true,
            speed: 14,
            bounciness: 6,
          }).start();
        }, 200);
        Animated.timing(confidenceBarScale, {
          toValue: res.confidence,
          duration: duration(800),
          useNativeDriver: true,
        }).start();
      })
      .catch((err) => {
        setIsAnalyzing(false);
        const message =
          launchedFromShare && (err.message || "").toLowerCase().includes("invalid")
            ? "Failed to load shared image."
            : err.message || "Verification failed. Please try again.";
        setScanError(message);
      });

    return () => bobbing?.stop?.();
  }, [selectedUri, reduceMotion]);

  useEffect(() => {
    if (!isAnalyzing) return;
    const loop = Animated.loop(
      Animated.sequence([
        Animated.timing(analyzingPulse, {
          toValue: 1,
          duration: 400,
          useNativeDriver: true,
        }),
        Animated.timing(analyzingPulse, {
          toValue: 0,
          duration: 400,
          useNativeDriver: true,
        }),
      ])
    );
    loop.start();
    return () => loop.stop();
  }, [isAnalyzing, analyzingPulse]);

  useEffect(() => {
    if (!isAnalyzing) return;
    analyzingShimmer.setValue(0);
    const shimmerLoop = Animated.loop(
      Animated.sequence([
        Animated.timing(analyzingShimmer, {
          toValue: 1,
          duration: 1200,
          useNativeDriver: true,
        }),
        Animated.timing(analyzingShimmer, {
          toValue: 0,
          duration: 0,
          useNativeDriver: true,
        }),
      ])
    );
    shimmerLoop.start();
    return () => shimmerLoop.stop();
  }, [isAnalyzing]);

  const onPressIn = () => {
    Animated.parallel([
      Animated.spring(pressAnim, {
        toValue: 0.97,
        useNativeDriver: true,
        speed: 22,
        bounciness: 0,
      }),
      Animated.timing(buttonGlowPress, {
        toValue: 1,
        duration: 100,
        useNativeDriver: false,
      }),
    ]).start();
  };

  const onPressOut = () => {
    Animated.parallel([
      Animated.spring(pressAnim, {
        toValue: 1,
        useNativeDriver: true,
        speed: 18,
        bounciness: 6,
      }),
      Animated.timing(buttonGlowPress, {
        toValue: 0,
        duration: 250,
        useNativeDriver: false,
      }),
    ]).start();
  };

  const pickImage = async () => {
    setMessage("");

    const perm = await ensureGalleryPermission();
    if (!perm.ok) {
      setMessage(perm.message);
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ["images"],
      allowsMultipleSelection: false,
      allowsEditing: false,
      quality: 1,
    });

    if (result.canceled) {
      setMessage("Selection cancelled.");
      return;
    }

    const asset = result.assets?.[0];
    if (!asset?.uri) {
      setMessage("Could not read selected image.");
      return;
    }

    setSelectedUri(asset.uri);
    setMessage("Image selected. Analyzing…");
  };

  const clearImage = () => {
    setSelectedUri(null);
    setMessage("");
    setScanError(null);
  };

  const handleDoneFromShare = () => {
    if (Platform.OS === "android") {
      BackHandler.exitApp();
    } else {
      clearImage();
      setLaunchedFromShare(false);
    }
  };

  if (!fontsLoaded) {
    return (
      <SafeAreaProvider>
        <SafeAreaView
          edges={["top", "bottom"]}
          style={[styles.safe, { backgroundColor: COLORS.bgMain }]}
        >
          <StatusBar barStyle="light-content" />
          <View style={styles.loadingWrap}>
            <Animated.View style={styles.loadingDot} />
            <Text style={styles.loadingText}>Loading…</Text>
          </View>
        </SafeAreaView>
      </SafeAreaProvider>
    );
  }

  return (
    <SafeAreaProvider>
      <SafeAreaView edges={["top", "bottom"]} style={styles.safe}>
        <StatusBar barStyle="light-content" />

      <LinearGradient
        colors={[COLORS.bgMain, COLORS.bgSoft, COLORS.bgDeep]}
        style={StyleSheet.absoluteFill}
      />
      <View pointerEvents="none" style={styles.textureOverlay} />
      <View style={styles.gridLines}>
        {[0, 1, 2, 3, 4, 5, 6, 7, 8].map((i) => (
          <View key={`h${i}`} style={[styles.gridLine, styles.gridLineH, { top: `${11.11 * i}%` }]} />
        ))}
        {[0, 1, 2, 3, 4, 5, 6, 7, 8].map((i) => (
          <View key={`v${i}`} style={[styles.gridLine, styles.gridLineV, { left: `${11.11 * i}%` }]} />
        ))}
      </View>

      {!reduceMotion && (
        <View style={styles.particlesWrap} pointerEvents="none">
          {particleAnims.map((anim, i) => (
            <Animated.View
              key={i}
              style={[
                styles.particle,
                {
                  left: `${(i * 5 + 2) % 98}%`,
                  top: `${(i * 8 + 3) % 98}%`,
                  opacity: 0.45 + (i % 4) * 0.12,
                  transform: [
                    {
                      translateY: anim.interpolate({
                        inputRange: [0, 1],
                        outputRange: [0, -280],
                      }),
                    },
                  ],
                },
              ]}
            />
          ))}
        </View>
      )}

      <Animated.ScrollView
        style={styles.scroll}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
        keyboardShouldPersistTaps="handled"
        scrollEventThrottle={16}
        onScroll={Animated.event(
          [{ nativeEvent: { contentOffset: { y: scrollY } } }],
          { useNativeDriver: false }
        )}
      >
        <View style={styles.page}>
        <View style={styles.hero}>
          <Animated.View
            style={[
              styles.heroPanel,
              {
                opacity: heroPanelOpacity,
                transform: [
                  {
                    translateY: scrollY.interpolate({
                      inputRange: [0, 180],
                      outputRange: [0, -24],
                      extrapolate: "clamp",
                    }),
                  },
                ],
              },
            ]}
          >
          <Animated.View
            style={{
              transform: [
                { perspective: 400 },
                { rotateX: heroTitleRotateX.interpolate({ inputRange: [-12, 0], outputRange: ["-12deg", "0deg"] }) },
                { scale: heroTitleScale },
                { translateY: heroTitleTranslateY },
              ],
            }}
          >
            <Animated.Text
              style={[
                styles.title,
                {
                  opacity: heroTitleOpacity,
                  textShadowColor: COLORS.glowBlue,
                  textShadowOffset: { width: 0, height: 0 },
                  ...(Platform.OS === "ios" && {
                    textShadowRadius: heroGlowFlicker.interpolate({
                      inputRange: [0, 1],
                      outputRange: [8, 20],
                    }),
                  }),
                  ...(Platform.OS === "android" && { textShadowRadius: 14 }),
                },
              ]}
              accessibilityRole="header"
            >
              UNMASK
            </Animated.Text>
          </Animated.View>
          <Animated.Text style={[styles.heroSub, { opacity: heroTaglineOpacity }]}>
            "Expose manipulated media"
          </Animated.Text>
          <Animated.Text style={[styles.tagline, { opacity: heroTaglineOpacity }]}>
            Verify images responsibly and ethically
          </Animated.Text>

          <View style={styles.buttons}>
            <Animated.View
              style={[
                styles.primaryButtonWrap,
                Platform.OS === "ios" && {
                  shadowColor: COLORS.glowBlue,
                  shadowOffset: { width: 0, height: 8 },
                  shadowOpacity: buttonShadow.shadowOpacity,
                  shadowRadius: buttonShadow.shadowRadius,
                },
                Platform.OS === "android" && { elevation: 14 },
              ]}
            >
              <Animated.View style={{ transform: [{ scale: pressAnim }] }}>
                <Pressable
                  onPress={pickImage}
                  onPressIn={onPressIn}
                  onPressOut={onPressOut}
                  style={styles.primaryButton}
                  accessibilityLabel="Verify media"
                  accessibilityRole="button"
                >
                  <Text style={styles.primaryButtonText}>VERIFY MEDIA</Text>
                </Pressable>
              </Animated.View>
            </Animated.View>

            <Animated.View style={{ transform: [{ scale: pressAnim }] }}>
              <Pressable
                onPress={() => setAboutVisible(true)}
                onPressIn={onPressIn}
                onPressOut={onPressOut}
                style={styles.secondaryButton}
                accessibilityLabel="About Unmask"
                accessibilityRole="button"
              >
                <Text style={styles.secondaryButtonText}>ABOUT</Text>
              </Pressable>
            </Animated.View>
          </View>

          {message ? (
            <Text style={styles.message} accessibilityLiveRegion="polite">
              {message}
            </Text>
          ) : null}
          {shareError && !selectedUri ? (
            <View style={styles.shareErrorBanner}>
              <Text style={styles.shareErrorBannerText}>{shareError}</Text>
            </View>
          ) : null}
          </Animated.View>
        </View>

        {shareBannerMessage && selectedUri ? (
          <View style={styles.shareBanner}>
            <Text style={styles.shareBannerText}>{shareBannerMessage}</Text>
          </View>
        ) : null}

        {selectedUri ? (
          <Animated.View
            style={[
              styles.verifyCardWrap,
              {
                opacity: previewFade,
                transform: [
                  {
                    scale: previewFade.interpolate({
                      inputRange: [0, 1],
                      outputRange: [0.96, 1],
                    }),
                  },
                ],
              },
            ]}
          >
            <View style={styles.verifyCardDepth} />
            <View style={styles.verifyCard}>
              <Animated.View
                style={[
                  styles.imageContainer,
                  {
                    transform: [
                      { translateY: imageTranslateY },
                      {
                        translateY: imageBob.interpolate({
                          inputRange: [0, 1],
                          outputRange: [0, 5],
                        }),
                      },
                    ],
                  },
                ]}
              >
                <Image
                  source={{ uri: selectedUri }}
                  style={styles.previewImage}
                  resizeMode="contain"
                  accessibilityLabel="Selected image for verification"
                />
                <View style={styles.imageGlowBorder} />
              </Animated.View>

              {isAnalyzing ? (
                <View style={styles.analyzingRow}>
                  <View style={styles.analyzingShimmerWrap}>
                    <Animated.View
                      style={[
                        styles.analyzingShimmerBar,
                        {
                          opacity: analyzingShimmer.interpolate({
                            inputRange: [0, 0.5, 1],
                            outputRange: [0.2, 0.8, 0.2],
                          }),
                          transform: [
                            {
                              translateX: analyzingShimmer.interpolate({
                                inputRange: [0, 1],
                                outputRange: [-SCREEN_WIDTH, SCREEN_WIDTH],
                              }),
                            },
                          ],
                        },
                      ]}
                    />
                  </View>
                  <View style={styles.loadingDots}>
                    {[0, 1, 2].map((i) => (
                      <Animated.View
                        key={i}
                        style={[
                          styles.analyzingDot,
                          {
                            opacity: analyzingPulse.interpolate({
                              inputRange: [0, 0.33, 0.66, 1],
                              outputRange:
                                i === 0
                                  ? [0.4, 1, 0.4, 0.4]
                                  : i === 1
                                    ? [0.4, 0.4, 1, 0.4]
                                    : [0.4, 0.4, 0.4, 1],
                            }),
                          },
                        ]}
                      />
                    ))}
                  </View>
                  <Text style={styles.analyzingText}>Analyzing…</Text>
                </View>
              ) : verification ? (
                <Animated.View
                  style={[
                    styles.resultBlock,
                    {
                      opacity: resultShimmer,
                      transform: [
                        {
                          scale: resultShimmer.interpolate({
                            inputRange: [0, 1],
                            outputRange: [0.98, 1],
                          }),
                        },
                      ],
                    },
                  ]}
                >
                  <Animated.View
                    style={[
                      styles.resultBadge,
                      verification.label === "Likely real"
                        ? styles.resultBadgeReal
                        : styles.resultBadgeAi,
                      { transform: [{ scale: resultBadgePulse }] },
                    ]}
                  >
                    <Text
                      style={[
                        styles.resultBadgeText,
                        verification.label === "Likely real"
                          ? styles.resultBadgeTextReal
                          : styles.resultBadgeTextAi,
                      ]}
                    >
                      {verification.label}
                    </Text>
                  </Animated.View>

                  <View style={styles.confidenceBarWrap}>
                    <Animated.View
                      style={[
                        styles.confidenceBarFillWrap,
                        {
                          transform: [{ scaleX: confidenceBarScale }],
                        },
                      ]}
                    >
                      <LinearGradient
                        colors={[COLORS.accentCyan, COLORS.accentBlue]}
                        start={{ x: 0, y: 0 }}
                        end={{ x: 1, y: 0 }}
                        style={styles.confidenceBarFill}
                      />
                    </Animated.View>
                  </View>
                  <Text style={styles.confidenceLabel}>
                    Confidence: {Math.round((verification.confidence ?? 0) * 100)}%
                  </Text>

                  <Text style={styles.explanation}>{verification.explanation}</Text>

                  <Text style={styles.disclaimer}>
                    {verification.disclaimer ?? "AI detection is not 100% accurate. Use results as one input for your judgment."}
                  </Text>
                </Animated.View>
              ) : null}

              {scanError ? (
                <View style={styles.errorBlock}>
                  <Text style={styles.errorText}>{scanError}</Text>
                </View>
              ) : null}

              <Pressable
                onPress={launchedFromShare ? handleDoneFromShare : clearImage}
                style={styles.clearButton}
                accessibilityLabel={launchedFromShare ? "Done" : "Choose another image"}
                accessibilityRole="button"
              >
                <Text style={styles.clearButtonText}>
                  {launchedFromShare ? "Done" : "Choose another image"}
                </Text>
              </Pressable>
            </View>
          </Animated.View>
        ) : (
          <View style={styles.placeholderCard}>
            <Text style={styles.placeholderTitle}>No image selected</Text>
            <Text style={styles.placeholderText}>
              Tap “Verify Media” to choose a photo from your gallery.
            </Text>
          </View>
        )}
        </View>
      </Animated.ScrollView>

        <Modal
          visible={aboutVisible}
          transparent
          animationType="fade"
          onRequestClose={() => setAboutVisible(false)}
        >
          <Pressable
            style={styles.modalBackdrop}
            onPress={() => setAboutVisible(false)}
            accessibilityLabel="Close about"
          >
            <View style={styles.modalContent}>
              <Text style={styles.modalTitle}>UNMASK</Text>
              <Text style={styles.modalTagline}>
                Verify media responsibly and ethically
              </Text>
              <Text style={styles.modalBody}>
                Unmask is a racial equity-centered deepfake verification tool designed 
                to fight bias in AI detection systems. It helps people verify whether 
                media is authentic or AI-generated. Our pipeline is built to improve 
                performance on underrepresented faces, especially Black and people of 
                color. From images to video and audio, we believe media verification 
                should be transparent, responsible, and accessible to everyone.
              </Text>
              <Pressable
                onPress={() => setAboutVisible(false)}
                style={styles.modalButton}
                accessibilityLabel="Close"
                accessibilityRole="button"
              >
                <Text style={styles.modalButtonText}>Close</Text>
              </Pressable>
            </View>
          </Pressable>
        </Modal>
      </SafeAreaView>
    </SafeAreaProvider>
  );
}

const styles = StyleSheet.create({
  safe: {
    flex: 1,
    backgroundColor: COLORS.bgMain,
  },
  textureOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: "transparent",
    opacity: 0.03,
  },
  gridLines: {
    ...StyleSheet.absoluteFillObject,
    pointerEvents: "none",
  },
  gridLine: {
    position: "absolute",
    backgroundColor: COLORS.accentCyan,
    opacity: 0.18,
  },
  particlesWrap: {
    ...StyleSheet.absoluteFillObject,
    overflow: "hidden",
  },
  particle: {
    position: "absolute",
    width: 5,
    height: 5,
    borderRadius: 2.5,
    backgroundColor: COLORS.accentCyan,
  },
  gridLineH: {
    left: 0,
    right: 0,
    height: 1,
  },
  gridLineV: {
    top: 0,
    bottom: 0,
    width: 1,
  },
  scroll: {
    flex: 1,
  },
  scrollContent: {
    paddingTop: SPACING * 2,
    paddingBottom: SPACING * 3,
  },
  page: {
    width: CONTENT_WIDTH_PERCENT,
    maxWidth: MAX_CONTENT_WIDTH,
    alignSelf: "center",
    alignItems: "center",
  },
  hero: {
    alignItems: "center",
    width: "100%",
    marginBottom: SPACING * 1.75,
  },
  heroPanel: {
    width: "100%",
    maxWidth: MAX_CONTENT_WIDTH,
    alignItems: "center",
    paddingVertical: SPACING * 2.25,
    paddingHorizontal: SPACING * 1.75,
    borderRadius: RADIUS + 6,
    backgroundColor: "rgba(255,255,255,0.72)",
    borderWidth: 1,
    borderColor: "rgba(10,28,255,0.18)",
    overflow: "hidden",
    ...(Platform.OS === "ios" && {
      shadowColor: COLORS.glowBlue,
      shadowOffset: { width: 0, height: 10 },
      shadowOpacity: 0.2,
      shadowRadius: 24,
    }),
    ...(Platform.OS === "android" && { elevation: 8 }),
  },
  title: {
    fontFamily: "Orbitron_700Bold",
    fontSize: 38,
    letterSpacing: 7.5,
    color: COLORS.textPrimary,
    textAlign: "center",
    flexShrink: 1,
  },
  heroSub: {
    fontFamily: "Exo2_600SemiBold",
    fontSize: 16,
    letterSpacing: 1.2,
    color: COLORS.textSecondary,
    marginTop: 8,
    textAlign: "center",
  },
  tagline: {
    fontFamily: "Inter_400Regular",
    fontSize: 14,
    color: COLORS.textMuted,
    marginTop: 10,
    textAlign: "center",
    maxWidth: 320,
    lineHeight: 21,
  },
  buttons: {
    marginTop: SPACING * 2,
    gap: SPACING - 2,
    alignItems: "center",
    width: "100%",
    maxWidth: 360,
  },
  primaryButtonWrap: {
    borderRadius: 999,
    shadowColor: COLORS.glowBlue,
    width: "100%",
  },
  primaryButton: {
    height: BUTTON_HEIGHT,
    paddingHorizontal: SPACING * 1.25,
    borderRadius: 999,
    borderWidth: 2,
    borderColor: COLORS.accentBlue,
    backgroundColor: "#ffffff",
    width: "100%",
    alignItems: "center",
    justifyContent: "center",
    ...(Platform.OS === "ios" && {
      shadowColor: COLORS.glowBlue,
      shadowOffset: { width: 0, height: 2 },
      shadowOpacity: 0.2,
      shadowRadius: 6,
    }),
  },
  primaryButtonText: {
    fontFamily: "Exo2_600SemiBold",
    fontSize: 15,
    letterSpacing: 1.6,
    color: COLORS.accentBlue,
  },
  secondaryButton: {
    height: BUTTON_HEIGHT,
    paddingHorizontal: SPACING * 1.25,
    borderRadius: 999,
    borderWidth: 2,
    borderColor: COLORS.accentBlue,
    backgroundColor: "rgba(255,255,255,0.65)",
    width: "100%",
    alignItems: "center",
    justifyContent: "center",
    ...(Platform.OS === "ios" && {
      shadowColor: COLORS.cardShadow,
      shadowOffset: { width: 0, height: 4 },
      shadowOpacity: 0.1,
      shadowRadius: 8,
    }),
    ...(Platform.OS === "android" && { elevation: 3 }),
  },
  secondaryButtonText: {
    fontFamily: "Inter_500Medium",
    fontSize: 14,
    letterSpacing: 0.8,
    color: COLORS.accentBlue,
  },
  message: {
    marginTop: SPACING - 2,
    minHeight: 22,
    fontFamily: "Inter_400Regular",
    fontSize: 13,
    color: COLORS.textSecondary,
    textAlign: "center",
    width: "100%",
  },
  shareBanner: {
    width: "100%",
    maxWidth: MAX_CONTENT_WIDTH,
    marginBottom: SPACING - 4,
    paddingVertical: 10,
    paddingHorizontal: 16,
    borderRadius: 12,
    backgroundColor: "rgba(0,240,255,0.12)",
    borderWidth: 1,
    borderColor: "rgba(0,240,255,0.25)",
  },
  shareBannerText: {
    fontFamily: "Inter_500Medium",
    fontSize: 13,
    color: COLORS.textOnDarkSecondary,
    textAlign: "center",
  },
  shareErrorBanner: {
    width: "100%",
    maxWidth: 380,
    marginTop: 14,
    paddingVertical: 12,
    paddingHorizontal: 16,
    borderRadius: 12,
    backgroundColor: "rgba(239,68,68,0.12)",
    borderWidth: 1,
    borderColor: "rgba(239,68,68,0.3)",
  },
  shareErrorBannerText: {
    fontFamily: "Inter_500Medium",
    fontSize: 13,
    color: "#b91c1c",
    textAlign: "center",
  },
  verifyCardWrap: {
    width: "100%",
    maxWidth: MAX_CONTENT_WIDTH,
    alignItems: "center",
    position: "relative",
  },
  verifyCardDepth: {
    position: "absolute",
    top: 8,
    left: 12,
    right: 12,
    bottom: -6,
    backgroundColor: COLORS.bgDeep,
    borderRadius: 28,
    opacity: 0.9,
    ...(Platform.OS === "android" && { elevation: 2 }),
  },
  verifyCard: {
    width: "100%",
    backgroundColor: "rgba(255,255,255,0.85)",
    borderRadius: RADIUS + 2,
    padding: SPACING + 4,
    alignItems: "center",
    borderWidth: 1,
    borderColor: "rgba(10,28,255,0.22)",
    ...(Platform.OS === "ios" && {
      shadowColor: COLORS.glowBlue,
      shadowOffset: { width: 0, height: 12 },
      shadowOpacity: 0.22,
      shadowRadius: 28,
    }),
    ...(Platform.OS === "android" && { elevation: 12 }),
  },
  imageContainer: {
    width: "100%",
    aspectRatio: 4 / 3,
    borderRadius: RADIUS - 6,
    overflow: "hidden",
    position: "relative",
    backgroundColor: COLORS.bgDeep,
    borderWidth: 1,
    borderColor: "rgba(10,28,255,0.12)",
  },
  imageGlowBorder: {
    ...StyleSheet.absoluteFillObject,
    borderRadius: RADIUS - 6,
    borderWidth: 2,
    borderColor: "rgba(0,240,255,0.4)",
    pointerEvents: "none",
  },
  previewImage: {
    width: "100%",
    height: "100%",
  },
  analyzingRow: {
    marginTop: 20,
    alignItems: "center",
    gap: 10,
  },
  analyzingShimmerWrap: {
    position: "absolute",
    top: -8,
    left: -24,
    right: -24,
    height: 4,
    overflow: "hidden",
    borderRadius: 2,
  },
  analyzingShimmerBar: {
    position: "absolute",
    left: 0,
    top: 0,
    width: 80,
    height: 4,
    borderRadius: 2,
    backgroundColor: COLORS.accentCyan,
  },
  loadingDots: {
    flexDirection: "row",
    gap: 8,
  },
  analyzingDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: COLORS.accentCyanDark,
    opacity: 0.9,
  },
  analyzingText: {
    fontFamily: "Inter_500Medium",
    fontSize: 14,
    color: COLORS.textSecondary,
  },
  resultBlock: {
    width: "100%",
    marginTop: 20,
    alignItems: "center",
  },
  resultBadge: {
    paddingVertical: 10,
    paddingHorizontal: 20,
    borderRadius: 12,
    borderWidth: 2,
  },
  resultBadgeReal: {
    borderColor: "rgba(34,197,94,0.6)",
    backgroundColor: "rgba(34,197,94,0.12)",
    shadowColor: COLORS.realGlow,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.8,
    shadowRadius: 12,
  },
  resultBadgeAi: {
    borderColor: "rgba(239,68,68,0.6)",
    backgroundColor: "rgba(239,68,68,0.12)",
    shadowColor: COLORS.aiGlow,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.8,
    shadowRadius: 12,
  },
  resultBadgeText: {
    fontFamily: "Exo2_600SemiBold",
    fontSize: 14,
    letterSpacing: 1,
  },
  resultBadgeTextReal: {
    color: "#166534",
  },
  resultBadgeTextAi: {
    color: "#b91c1c",
  },
  confidenceBarWrap: {
    width: "100%",
    height: 8,
    borderRadius: 4,
    backgroundColor: COLORS.bgSoft,
    marginTop: 18,
    overflow: "hidden",
  },
  confidenceBarFillWrap: {
    width: "100%",
    height: "100%",
    borderRadius: 4,
    overflow: "hidden",
  },
  confidenceBarFill: {
    width: "100%",
    height: "100%",
    borderRadius: 4,
  },
  confidenceLabel: {
    fontFamily: "Inter_500Medium",
    fontSize: 12,
    color: COLORS.textSecondary,
    marginTop: 8,
  },
  explanation: {
    fontFamily: "Inter_400Regular",
    fontSize: 13,
    color: COLORS.textSecondary,
    textAlign: "center",
    marginTop: 14,
    lineHeight: 20,
    paddingHorizontal: 8,
  },
  disclaimer: {
    fontFamily: "Inter_400Regular",
    fontSize: 11,
    color: COLORS.textMuted,
    textAlign: "center",
    marginTop: 12,
    lineHeight: 16,
    paddingHorizontal: 4,
  },
  errorBlock: {
    width: "100%",
    marginTop: 16,
    paddingVertical: 12,
    paddingHorizontal: 16,
    borderRadius: 12,
    backgroundColor: "rgba(239,68,68,0.12)",
    borderWidth: 1,
    borderColor: "rgba(239,68,68,0.3)",
  },
  errorText: {
    fontFamily: "Inter_400Regular",
    fontSize: 13,
    color: "#b91c1c",
    textAlign: "center",
    lineHeight: 19,
  },
  clearButton: {
    marginTop: 20,
    paddingVertical: 12,
    paddingHorizontal: 20,
  },
  clearButtonText: {
    fontFamily: "Inter_500Medium",
    fontSize: 13,
    color: COLORS.accentBlueDark,
  },
  placeholderCard: {
    width: "100%",
    maxWidth: MAX_CONTENT_WIDTH,
    paddingVertical: SPACING * 2.75,
    paddingHorizontal: SPACING * 1.5,
    borderRadius: RADIUS + 2,
    borderWidth: 1,
    borderColor: "rgba(37,99,235,0.18)",
    backgroundColor: COLORS.bgSoft,
    alignItems: "center",
    gap: 10,
    ...(Platform.OS === "ios" && {
      shadowColor: COLORS.cardShadow,
      shadowOffset: { width: 0, height: 4 },
      shadowOpacity: 0.08,
      shadowRadius: 12,
    }),
    ...(Platform.OS === "android" && { elevation: 4 }),
  },
  placeholderTitle: {
    fontFamily: "Exo2_600SemiBold",
    fontSize: 15,
    color: COLORS.textPrimary,
  },
  placeholderText: {
    fontFamily: "Inter_400Regular",
    fontSize: 13,
    color: COLORS.textMuted,
    textAlign: "center",
    lineHeight: 20,
  },
  loadingWrap: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    gap: 16,
  },
  loadingDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    backgroundColor: COLORS.accentCyan,
  },
  loadingText: {
    fontFamily: "Inter_400Regular",
    fontSize: 14,
    color: COLORS.textOnDarkSecondary,
  },
  modalBackdrop: {
    flex: 1,
    backgroundColor: "rgba(11,26,51,0.4)",
    justifyContent: "center",
    alignItems: "center",
    padding: 24,
  },
  modalContent: {
    width: "100%",
    maxWidth: 340,
    backgroundColor: COLORS.card,
    borderRadius: 24,
    padding: 28,
    alignItems: "center",
    borderWidth: 1,
    borderColor: "rgba(8,145,168,0.2)",
    ...(Platform.OS === "ios" && {
      shadowColor: COLORS.accentCyanDark,
      shadowOffset: { width: 0, height: 16 },
      shadowOpacity: 0.22,
      shadowRadius: 32,
    }),
    ...(Platform.OS === "android" && { elevation: 16 }),
  },
  modalTitle: {
    fontFamily: "Orbitron_700Bold",
    fontSize: 26,
    letterSpacing: 4,
    color: COLORS.textPrimary,
  },
  modalTagline: {
    fontFamily: "Inter_500Medium",
    fontSize: 14,
    color: COLORS.accentBlueDark,
    marginTop: 10,
    textAlign: "center",
  },
  modalBody: {
    fontFamily: "Inter_400Regular",
    fontSize: 14,
    color: COLORS.textSecondary,
    textAlign: "center",
    marginTop: 16,
    lineHeight: 22,
  },
  modalButton: {
    marginTop: 24,
    paddingVertical: 14,
    paddingHorizontal: 32,
    borderRadius: 999,
    backgroundColor: COLORS.bgSoft,
    borderWidth: 1.5,
    borderColor: COLORS.accentBlue,
  },
  modalButtonText: {
    fontFamily: "Inter_500Medium",
    fontSize: 14,
    color: COLORS.accentBlue,
  },
});
