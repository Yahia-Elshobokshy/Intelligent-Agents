const functions = require("firebase-functions");
const admin = require("firebase-admin");
admin.initializeApp();

const db = admin.firestore();

// Helper: get all FCM tokens for a house
async function getHouseTokens(houseId) {
  const snapshot = await db.collection("users")
    .where("houseId", "==", houseId)
    .get();

  const tokens = [];
  snapshot.forEach(doc => {
    const userTokens = doc.data().fcmTokens || [];
    tokens.push(...userTokens);
  });
  return [...new Set(tokens)]; // deduplicate
}

// Helper: send to multiple tokens
async function sendToTokens(tokens, title, body, data = {}) {
  if (tokens.length === 0) return;
  const message = {
    notification: { title, body },
    data,
    tokens,
    android: { priority: "high" },
    apns: { payload: { aps: { sound: "default" } } },
  };
  return admin.messaging().sendEachForMulticast(message);
}

// --- TRIGGER 1: Gate opened or closed ---
exports.onGateStatusChange = functions.firestore
  .document("houses/{houseId}/gates/{gateId}")
  .onUpdate(async (change, context) => {
    const before = change.before.data();
    const after = change.after.data();
    const { houseId, gateId } = context.params;

    if (before.status === after.status) return; // no change

    const gateName = after.name || gateId;
    const isOpen = after.status === "open";
    const triggeredBy = after.lastTriggeredBy || "System";

    const title = isOpen ? "🔓 Gate Opened" : "🔒 Gate Closed";
    const body = `${gateName} was ${isOpen ? "opened" : "closed"} by ${triggeredBy}`;

    const tokens = await getHouseTokens(houseId);
    await sendToTokens(tokens, title, body, {
      type: "gate_status",
      gateId,
      status: after.status,
    });
  });

// --- TRIGGER 2: OTP used ---
exports.onOTPUsed = functions.firestore
  .document("houses/{houseId}/otps/{otpId}")
  .onUpdate(async (change, context) => {
    const before = change.before.data();
    const after = change.after.data();
    const { houseId } = context.params;

    if (!after.usedAt || before.usedAt) return; // only fire once on first use

    const title = "🔑 Guest Access Used";
    const body = `OTP ${after.code} was used to open ${after.gateName || "a gate"}`;

    const tokens = await getHouseTokens(houseId);
    await sendToTokens(tokens, title, body, { type: "otp_used" });
  });

// --- TRIGGER 3: Failed OTP attempt ---
exports.onFailedAttempt = functions.firestore
  .document("houses/{houseId}/failed_attempts/{attemptId}")
  .onCreate(async (snap, context) => {
    const { houseId } = context.params;
    const data = snap.data();

    const title = "⚠️ Failed Access Attempt";
    const body = data.reason === "wrong_otp"
      ? `Someone entered an invalid code at ${data.gateName || "your gate"}`
      : `Unauthorized access attempt detected at ${data.gateName || "your gate"}`;

    const tokens = await getHouseTokens(houseId);
    await sendToTokens(tokens, title, body, { type: "failed_attempt" });
  });

// --- TRIGGER 4: Any gate activity (camera detection) ---
exports.onGateActivity = functions.firestore
  .document("houses/{houseId}/access_logs/{logId}")
  .onCreate(async (snap, context) => {
    const { houseId } = context.params;
    const log = snap.data();

    // Only notify for camera-triggered events, not manual app actions
    if (log.triggeredBy !== "camera") return;

    const title = "📷 Gate Activity Detected";
    const body = `Motion/face detected at ${log.gateName || "your gate"}`;

    const tokens = await getHouseTokens(houseId);
    await sendToTokens(tokens, title, body, { type: "camera_activity" });
  });