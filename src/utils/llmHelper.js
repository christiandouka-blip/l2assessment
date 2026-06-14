import Groq from 'groq-sdk';

/**
 * LLM Helper for categorizing customer support messages
 * Using Groq API for AI-powered categorization
 */

// Initialize Groq client
const groq = new Groq({
  apiKey: import.meta.env.VITE_GROQ_API_KEY,
  dangerouslyAllowBrowser: true // Required for browser-based calls (not recommended for production!)
});

/**
 * Categorize a customer support message using Groq AI
 *
 * Returns a structured triage object instead of freeform text so the rest of
 * the app can rely on consistent fields.
 *
 * @param {string} message - The customer support message
 * @returns {Promise<{category: string, urgency: string, route: string, confidence: number, reasoning: string, recommendedAction: string}>}
 */
export async function categorizeMessage(message) {
  try {
    const response = await groq.chat.completions.create({
      model: "llama-3.3-70b-versatile",
      messages: [
        {
          role: "system",
          content: `You are an AI customer support triage assistant. Analyze the customer message and respond with a SINGLE JSON object ONLY. Do not include markdown, code fences, or any text outside the JSON.

The JSON must match this exact schema:
{
  "category": "Billing Issue | Technical Problem | Feature Request | General Inquiry | Positive Feedback | Human Review",
  "secondaryCategory": "null OR one of the category values",
  "urgency": "High | Medium | Low",
  "route": "Billing Team | Technical Support | Product Team | Customer Success | Human Review",
  "confidence": 0.0,
  "reasoning": "short explanation",
  "recommendedAction": "next step"
}

Rules:
- "category" must be exactly one of the listed values.
- "route" must be exactly one of the listed values and should align with the category: Billing Issue -> Billing Team, Technical Problem -> Technical Support, Feature Request -> Product Team, Positive Feedback -> Customer Success, General Inquiry -> Customer Success, Human Review -> Human Review.
- "urgency" reflects business impact and time-sensitivity (outages, double charges, refunds are High; suggestions and thanks are Low).
- "confidence" is a number between 0 and 1 describing how certain you are.
- "reasoning" is one or two short sentences.
- "recommendedAction" is the concrete next step for the assigned team.
- Multi-issue handling: if the message contains more than one distinct issue type (for example both Billing Issue and Technical Problem), set "category" to the most important issue, set "secondaryCategory" to the second issue, set "route" to "Human Review", set "confidence" to 0.75, and make "reasoning" explain that the message contains multiple issue types.
- If the message contains only a single issue type, set "secondaryCategory" to null.
- Return JSON only.`
        },
        {
          role: "user",
          content: message
        }
      ],
      temperature: 0.2,
      response_format: { type: "json_object" },
    });

    const content = response.choices[0].message.content;
    const parsed = JSON.parse(content);

    return normalizeTriage(parsed, message);
  } catch (error) {
    console.warn('Groq API failed, using mock response:', error.message);
    return getMockCategorization(message);
  }
}

/**
 * Validate and fill in any missing fields on the model's structured output,
 * falling back to the rule-based mock for anything the model omitted.
 */
function normalizeTriage(parsed, message) {
  const fallback = getMockCategorization(message);
  const data = parsed && typeof parsed === 'object' ? parsed : {};

  let confidence = Number(data.confidence);
  if (!Number.isFinite(confidence)) {
    confidence = fallback.confidence;
  }
  confidence = Math.min(1, Math.max(0, confidence));

  const category = data.category || fallback.category;
  const secondaryCategory = sanitizeSecondary(data.secondaryCategory, category);

  const result = {
    category,
    secondaryCategory,
    urgency: data.urgency || fallback.urgency,
    route: data.route || fallback.route,
    confidence,
    reasoning: data.reasoning || fallback.reasoning,
    recommendedAction: data.recommendedAction || fallback.recommendedAction,
  };

  // Enforce multi-issue rules consistently, regardless of what the model returned.
  if (secondaryCategory) {
    result.route = "Human Review";
    result.confidence = 0.75;
    result.reasoning = `This message contains multiple issue types (${category} and ${secondaryCategory}). Routing to Human Review so a person can address both.`;
  }

  return result;
}

/**
 * Returns a valid secondary category string, or null when the model did not
 * report a meaningful second issue.
 */
function sanitizeSecondary(value, category) {
  if (!value || typeof value !== 'string') return null;
  const trimmed = value.trim();
  if (!trimmed) return null;
  const lowered = trimmed.toLowerCase();
  if (lowered === 'none' || lowered === 'n/a' || lowered === 'null') return null;
  if (trimmed === category) return null;
  return trimmed;
}

/**
 * Default routing, urgency and recommended action per category, used so the
 * mock fallback can produce the same structured shape as the AI response.
 */
const ROUTE_BY_CATEGORY = {
  "Billing Issue": "Billing Team",
  "Technical Problem": "Technical Support",
  "Feature Request": "Product Team",
  "Positive Feedback": "Customer Success",
  "General Inquiry": "Customer Success",
  "Human Review": "Human Review",
};

const URGENCY_BY_CATEGORY = {
  "Billing Issue": "High",
  "Technical Problem": "High",
  "Feature Request": "Low",
  "Positive Feedback": "Low",
  "General Inquiry": "Medium",
  "Human Review": "Medium",
};

const ACTION_BY_CATEGORY = {
  "Billing Issue": "Route to the Billing Team to review the charges and process any refund.",
  "Technical Problem": "Escalate to Technical Support to investigate and restore service.",
  "Feature Request": "Share with the Product Team for roadmap consideration.",
  "Positive Feedback": "Forward to Customer Success to acknowledge and thank the customer.",
  "General Inquiry": "Respond with the relevant FAQ or knowledge-base link.",
  "Human Review": "Flag for manual human review.",
};

/**
 * Build a complete structured triage object from a single detected category.
 */
function buildTriage(category, reasoning, confidence) {
  return {
    category,
    secondaryCategory: null,
    urgency: URGENCY_BY_CATEGORY[category] || "Medium",
    route: ROUTE_BY_CATEGORY[category] || "Human Review",
    confidence,
    reasoning,
    recommendedAction: ACTION_BY_CATEGORY[category] || "Review manually.",
  };
}

const URGENCY_RANK = { High: 3, Medium: 2, Low: 1 };

/**
 * Pick the more severe of two urgency levels.
 */
function mostUrgent(a, b) {
  return (URGENCY_RANK[a] || 0) >= (URGENCY_RANK[b] || 0) ? a : b;
}

/**
 * Build a structured triage object for a message that contains more than one
 * issue type. These are always sent to Human Review with reduced confidence.
 */
function buildMultiIssueTriage(primary, secondary) {
  return {
    category: primary,
    secondaryCategory: secondary,
    urgency: mostUrgent(URGENCY_BY_CATEGORY[primary], URGENCY_BY_CATEGORY[secondary]),
    route: "Human Review",
    confidence: 0.75,
    reasoning: `This message contains multiple issue types (${primary} and ${secondary}). Routing to Human Review so a person can address both.`,
    recommendedAction: "Route to Human Review to triage and resolve both issues with the appropriate teams.",
  };
}

/**
 * Mock categorization for when API is unavailable
 */
function getMockCategorization(message) {
  const lowerMessage = message.toLowerCase();
  
  // Array of possible reasoning variations for each category
  const reasoningVariations = {
    billing: [
      "Based on keywords related to payments and billing, this appears to be a billing-related inquiry. The customer may need assistance with account charges or payment issues.",
      "This message contains billing terminology. The customer is likely experiencing issues with payments, invoices, or account charges.",
      "The message references financial matters related to the customer's account. This suggests a billing or payment concern that requires attention.",
    ],
    technical: [
      "This message describes technical difficulties or system errors. The customer is reporting functionality issues that may require engineering review.",
      "Based on error-related keywords, this appears to be a technical support issue. The customer is experiencing problems with product functionality.",
      "The message indicates a technical problem or bug. This requires investigation from the technical support team.",
      "System-related issues are mentioned in this message. The customer needs technical assistance to resolve functionality problems.",
    ],
    feature: [
      "This message suggests improvements or new functionality. The customer is providing product feedback and feature suggestions.",
      "The customer is requesting enhancements to the product. This appears to be a feature request that should be reviewed by the product team.",
      "Based on the language used, this seems to be a suggestion for product improvements rather than a support issue.",
    ],
    inquiry: [
      "This appears to be a general question about the product or service. The customer is seeking information or clarification.",
      "The message contains questions that don't indicate a specific problem. This is likely a general inquiry requiring informational support.",
      "Based on the question format, this seems to be an information request rather than a technical or billing issue.",
    ],
    positive: [
      "This message contains positive sentiment and appreciation. While not a support request, it may warrant acknowledgment.",
      "The customer is expressing satisfaction or gratitude. This doesn't appear to require immediate support action.",
    ],
    ambiguous: [
      "The message content is unclear or doesn't match standard support categories. Manual review may be needed for proper categorization.",
      "This message doesn't contain clear indicators for automatic categorization. Human review recommended.",
    ]
  };
  
  // Helper to get random reasoning
  const getRandomReasoning = (category) => {
    const reasons = reasoningVariations[category];
    return reasons[Math.floor(Math.random() * reasons.length)];
  };
  
  // Billing-related detection
  const isBilling = lowerMessage.includes('bill') || lowerMessage.includes('payment') || 
      lowerMessage.includes('charge') || lowerMessage.includes('invoice') ||
      lowerMessage.includes('credit card') || lowerMessage.includes('subscription') ||
      lowerMessage.includes('refund') || lowerMessage.includes('cancel') && lowerMessage.includes('account');
  
  // Technical problem detection
  const isTechnical = lowerMessage.includes('bug') || lowerMessage.includes('error') || 
      lowerMessage.includes('broken') || lowerMessage.includes('not working') ||
      lowerMessage.includes('crash') || lowerMessage.includes('down') || 
      lowerMessage.includes('server') || lowerMessage.includes('loading') ||
      lowerMessage.includes('slow') || lowerMessage.includes('issue') ||
      lowerMessage.includes('problem') && !lowerMessage.includes('no problem');
  
  // Feature request detection
  const isFeature = lowerMessage.includes('feature') ||
      lowerMessage.includes('add') && (lowerMessage.includes('please') || lowerMessage.includes('could') || lowerMessage.includes('can you')) ||
      lowerMessage.includes('dark mode') ||
      lowerMessage.includes('improve') || lowerMessage.includes('would like to see') ||
      lowerMessage.includes('suggestion') || lowerMessage.includes('wish') ||
      lowerMessage.includes('could you') && lowerMessage.includes('add') ||
      lowerMessage.includes('enhancement') || lowerMessage.includes('would be great');
  
  // Positive feedback detection
  const isPositive = (lowerMessage.includes('thank') || lowerMessage.includes('thanks') || lowerMessage.includes('appreciate')) &&
      !lowerMessage.includes('but') && !lowerMessage.includes('however');
  
  // Question/inquiry detection
  const isInquiry = lowerMessage.includes('how') || lowerMessage.includes('what') || 
      lowerMessage.includes('when') || lowerMessage.includes('where') ||
      lowerMessage.includes('can i') || lowerMessage.includes('is there') ||
      lowerMessage.includes('?');
  
  // Collect detected issue types in priority order (most important first).
  const detectedIssues = [];
  if (isBilling) detectedIssues.push("Billing Issue");
  if (isTechnical) detectedIssues.push("Technical Problem");
  if (isFeature) detectedIssues.push("Feature Request");
  
  // Multi-issue: more than one distinct issue type -> route to Human Review.
  if (detectedIssues.length >= 2) {
    return buildMultiIssueTriage(detectedIssues[0], detectedIssues[1]);
  }
  
  // Single-issue / existing behavior.
  if (isBilling) {
    return buildTriage("Billing Issue", getRandomReasoning('billing'), 0.85);
  }
  if (isTechnical) {
    return buildTriage("Technical Problem", getRandomReasoning('technical'), 0.85);
  }
  if (isFeature) {
    return buildTriage("Feature Request", getRandomReasoning('feature'), 0.8);
  }
  if (isPositive) {
    return buildTriage("Positive Feedback", getRandomReasoning('positive'), 0.85);
  }
  if (isInquiry) {
    return buildTriage("General Inquiry", getRandomReasoning('inquiry'), 0.6);
  }
  
  // Fallback for ambiguous messages
  return buildTriage("Human Review", getRandomReasoning('ambiguous'), 0.4);
}
