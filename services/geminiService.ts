import { GoogleGenAI, Type, Schema } from "@google/genai";
import { AnalysisResult, PoseFeedback, Gender, PoseStyle, PoseLandmarks } from '../types';

const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

const analysisSchema: Schema = {
  type: Type.OBJECT,
  properties: {
    environment: { type: Type.STRING, description: "Detected location (e.g. Bedroom, Office, Cafe)" },
    lighting: {
      type: Type.OBJECT,
      properties: {
        quality: { type: Type.STRING, enum: ['Excellent', 'Good', 'Fair', 'Poor'] },
        direction: { type: Type.STRING, description: "Where is the light coming from relative to user? (e.g. 'From left', 'Backlit', 'Front')" },
        suggestion: { type: Type.STRING, description: "Actionable advice to fix lighting (e.g. 'Turn 45 degrees right')" }
      },
      required: ['quality', 'direction', 'suggestion']
    },
    background: {
      type: Type.OBJECT,
      properties: {
        clutterLevel: { type: Type.STRING, enum: ['Clean', 'Moderate', 'Cluttered'] },
        suggestion: { type: Type.STRING, description: "Advice for background (e.g. 'Step away from wall', 'Clear the pile of clothes')" }
      },
      required: ['clutterLevel', 'suggestion']
    },
    suggestedPose: {
      type: Type.OBJECT,
      properties: {
        title: { type: Type.STRING, description: "Catchy name for the pose" },
        description: { type: Type.STRING, description: "Overall description of the pose" },
        difficulty: { type: Type.STRING, enum: ['Easy', 'Medium', 'Hard'] },
        steps: { 
          type: Type.ARRAY, 
          items: { type: Type.STRING },
          description: "List of 3-4 specific instructions (e.g. 'Chin up', 'Hand in pocket')" 
        }
      },
      required: ['title', 'description', 'difficulty', 'steps']
    }
  },
  required: ['environment', 'lighting', 'background', 'suggestedPose']
};

const feedbackSchema: Schema = {
  type: Type.OBJECT,
  properties: {
    score: { type: Type.INTEGER, description: "Score from 0 to 100 based on how well the user matches the requested pose" },
    matchStatus: { type: Type.STRING, enum: ['Perfect', 'Good', 'Needs Improvement'] },
    adjustments: { 
      type: Type.ARRAY, 
      items: { type: Type.STRING },
      description: "List of 2-3 specific, short corrections (e.g. 'Lift left elbow', 'Turn head slightly right', 'Smile more')" 
    }
  },
  required: ['score', 'matchStatus', 'adjustments']
};

const landmarksSchema: Schema = {
  type: Type.OBJECT,
  properties: {
    nose: { type: Type.OBJECT, properties: { x: { type: Type.NUMBER }, y: { type: Type.NUMBER } } },
    leftShoulder: { type: Type.OBJECT, properties: { x: { type: Type.NUMBER }, y: { type: Type.NUMBER } } },
    rightShoulder: { type: Type.OBJECT, properties: { x: { type: Type.NUMBER }, y: { type: Type.NUMBER } } },
    leftElbow: { type: Type.OBJECT, properties: { x: { type: Type.NUMBER }, y: { type: Type.NUMBER } } },
    rightElbow: { type: Type.OBJECT, properties: { x: { type: Type.NUMBER }, y: { type: Type.NUMBER } } },
    leftWrist: { type: Type.OBJECT, properties: { x: { type: Type.NUMBER }, y: { type: Type.NUMBER } } },
    rightWrist: { type: Type.OBJECT, properties: { x: { type: Type.NUMBER }, y: { type: Type.NUMBER } } },
    leftHip: { type: Type.OBJECT, properties: { x: { type: Type.NUMBER }, y: { type: Type.NUMBER } } },
    rightHip: { type: Type.OBJECT, properties: { x: { type: Type.NUMBER }, y: { type: Type.NUMBER } } },
  },
};

export const analyzeSceneAndSuggest = async (
  base64Image: string,
  gender: Gender,
  style: PoseStyle
): Promise<AnalysisResult> => {
  try {
    const model = 'gemini-2.5-flash';
    const prompt = `
      Act as a world-class photographer and pose director.
      Analyze this image from a user's camera.
      
      User Gender: ${gender}
      Desired Vibe: ${style}

      1. **Analyze Environment & Background**: 
         - Where are they? 
         - Is the background messy? If so, suggest a pose that minimizes it (e.g., tight frame).
         - If the background is nice, suggest a pose that interacts with it.
      
      2. **Analyze Lighting**:
         - Detect the primary light source.
         - If the face is shadowed/backlit, tell them specifically which way to turn.
      
      3. **Suggest the Perfect Pose**:
         - Suggest a pose that matches the '${style}' vibe perfectly.
         - **Professional**: Confident, straight posture, trustworthy.
         - **Casual**: Relaxed, candid, natural movement.
         - **Creative**: Angular, using hands near face, interesting silhouette.
         - **Street**: Cool, leaning, wider stance.
         - **Glamour**: Chin down, eyes up, emphasize curves/lines.

      Return the result in JSON.
    `;

    const response = await ai.models.generateContent({
      model: model,
      contents: {
        parts: [
          { inlineData: { mimeType: 'image/jpeg', data: base64Image } },
          { text: prompt }
        ]
      },
      config: {
        responseMimeType: 'application/json',
        responseSchema: analysisSchema,
      }
    });

    if (response.text) {
      return JSON.parse(response.text) as AnalysisResult;
    }
    throw new Error("No analysis generated");
  } catch (error) {
    console.error("Analysis failed:", error);
    throw error;
  }
};

export const generatePoseReference = async (poseDescription: string, gender: Gender, style: PoseStyle): Promise<string> => {
  try {
    const model = 'gemini-2.5-flash-image';
    // We enhance the prompt to ensure the reference image is high contrast for the overlay
    const prompt = `
      Generate a reference photo for a pose.
      Model: ${gender}
      Vibe: ${style}
      Pose Description: ${poseDescription}
      
      CRITICAL OUTPUT REQUIREMENTS:
      1. Background: SOLID PITCH BLACK (#000000). This is mandatory for an overlay.
      2. Lighting: High contrast rim lighting or studio lighting.
      3. Clothing: Bright, solid colors (White, Neon, or Light Grey) to stand out against the black background.
      4. Shot: Full body or Upper body depending on the pose description.
      5. No text. Photorealistic.
    `;

    const response = await ai.models.generateContent({
      model: model,
      contents: {
        parts: [{ text: prompt }]
      },
      config: {
        // Nano Banana models don't support schema/json mime types
      }
    });

    const imagePart = response.candidates?.[0]?.content?.parts?.find(part => part.inlineData);
    if (imagePart?.inlineData?.data) {
      return imagePart.inlineData.data;
    }

    throw new Error("No image generated");
  } catch (error) {
    console.error("Image generation failed:", error);
    throw error;
  }
};

export const evaluatePoseMatch = async (
  base64Image: string,
  targetPoseDescription: string
): Promise<PoseFeedback> => {
  try {
    const model = 'gemini-2.5-flash';
    const prompt = `
      You are a strict pose coach.
      Target Pose: "${targetPoseDescription}"
      
      Analyze the user in the image.
      1. Compare their current body position to the Target Pose.
      2. Rate the accuracy (0-100).
      3. Give 3 specific, short commands to improve the pose (e.g. "Chin up", "Left hand in pocket", "Stand straighter").
      
      Return JSON.
    `;

    const response = await ai.models.generateContent({
      model: model,
      contents: {
        parts: [
          { inlineData: { mimeType: 'image/jpeg', data: base64Image } },
          { text: prompt }
        ]
      },
      config: {
        responseMimeType: 'application/json',
        responseSchema: feedbackSchema,
      }
    });

    if (response.text) {
      return JSON.parse(response.text) as PoseFeedback;
    }
    throw new Error("No feedback generated");
  } catch (error) {
    console.error("Evaluation failed:", error);
    throw error;
  }
};

export const extractPoseLandmarks = async (base64Image: string): Promise<PoseLandmarks | null> => {
  try {
    const model = 'gemini-2.5-flash';
    const prompt = `
      Analyze the pose in this image.
      Identify the approximate 2D coordinates (x, y) for the following body parts:
      nose, leftShoulder, rightShoulder, leftElbow, rightElbow, leftWrist, rightWrist, leftHip, rightHip.
      
      Coordinates should be normalized from 0 to 100.
      x=0 is left, x=100 is right.
      y=0 is top, y=100 is bottom.
    `;

    const response = await ai.models.generateContent({
      model: model,
      contents: {
        parts: [
          { inlineData: { mimeType: 'image/jpeg', data: base64Image } },
          { text: prompt }
        ]
      },
      config: {
        responseMimeType: 'application/json',
        responseSchema: landmarksSchema,
      }
    });

    if (response.text) {
      return JSON.parse(response.text) as PoseLandmarks;
    }
    return null;
  } catch (error) {
    console.error("Landmarks extraction failed:", error);
    return null;
  }
};
