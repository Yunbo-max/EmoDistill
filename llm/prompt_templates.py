"""
Dataset-specific prompt templates for different negotiation scenarios.

Env var EVOEMO_NO_EMOTION_PROMPT=1 strips the ``### EMOTIONAL APPROACH`` block
from creditor prompts. Used at eval time to test LoRA adapters trained on the
no-emotion sweep (so the training / eval prompt distribution matches).
"""

import os
import re
from typing import Dict, Any

_NEUTRAL_BLOCK = (
    "### NEGOTIATION STYLE:\nUse your best professional judgment to decide tone.\n"
)
_EMO_PATTERN = re.compile(
    r"### EMOTIONAL APPROACH\(you have to use the following emotion style if given\):\n"
    r".*?(?=\n###|\nRespond now|\nContinue the negotiation|\nYour response:|$)",
    flags=re.DOTALL,
)


def _maybe_strip_emotion(prompt: str) -> str:
    """Strip the EMOTIONAL APPROACH block when EVOEMO_NO_EMOTION_PROMPT is set."""
    if os.environ.get("EVOEMO_NO_EMOTION_PROMPT", "0") != "1":
        return prompt
    return _EMO_PATTERN.sub(_NEUTRAL_BLOCK, prompt, count=1)


class PromptTemplates:
    """
    Manages dataset-specific prompts for different negotiation contexts
    """
    
    @staticmethod
    def get_creditor_prompt(scenario_type: str, config: Dict[str, Any], emotion_config: Dict[str, Any],
                           timeline_text: str = "", debt_info: Dict[str, Any] = None) -> str:
        """
        Get creditor prompt based on scenario type.
        Honors EVOEMO_NO_EMOTION_PROMPT=1 to strip the emotion block at eval time.
        """
        if scenario_type == "debt":
            p = PromptTemplates._get_debt_creditor_prompt(config, emotion_config, timeline_text, debt_info)
        elif scenario_type == "disaster":
            p = PromptTemplates._get_disaster_creditor_prompt(config, emotion_config, timeline_text, debt_info)
        elif scenario_type == "student":
            p = PromptTemplates._get_student_creditor_prompt(config, emotion_config, timeline_text, debt_info)
        elif scenario_type == "medical":
            p = PromptTemplates._get_medical_creditor_prompt(config, emotion_config, timeline_text, debt_info)
        else:
            p = PromptTemplates._get_debt_creditor_prompt(config, emotion_config, timeline_text, debt_info)
        return _maybe_strip_emotion(p)
    
    @staticmethod
    def get_debtor_prompt(scenario_type: str, config: Dict[str, Any], emotion_prompt: str = "",
                         debt_info: Dict[str, Any] = None, timeline_text: str = "") -> str:
        """
        Get debtor prompt based on scenario type
        """
        if scenario_type == "debt":
            return PromptTemplates._get_debt_debtor_prompt(config, emotion_prompt, debt_info, timeline_text)
        elif scenario_type == "disaster":
            return PromptTemplates._get_disaster_debtor_prompt(config, emotion_prompt, debt_info, timeline_text)
        elif scenario_type == "student":
            return PromptTemplates._get_student_debtor_prompt(config, emotion_prompt, debt_info, timeline_text)
        elif scenario_type == "medical":
            return PromptTemplates._get_medical_debtor_prompt(config, emotion_prompt, debt_info, timeline_text)
        else:
            # Default to debt collection
            return PromptTemplates._get_debt_debtor_prompt(config, emotion_prompt, debt_info, timeline_text)
    
    # DEBT COLLECTION PROMPTS
    @staticmethod
    def _get_debt_creditor_prompt(config: Dict[str, Any], emotion_config: Dict[str, Any], 
                                 timeline_text: str, debt_info: Dict[str, Any]) -> str:
        outstanding_balance = debt_info.get('outstanding_balance', 0)
        return f"""You are a PROFESSIONAL Creditor debt collection agent negotiating payment timeline with the Debtor.

### CRITICAL NEGOTIATION RULES:
🚫 NEVER copy the debtor's exact number - this shows weakness
📉 Move GRADUALLY toward their position (not all at once) 
💪 Show you are negotiating, not just accepting
🎯 Your goal: Minimize payment days while reaching agreement

### ROLE CLARITY
- You are ONLY the Creditor - speak only as yourself
- Do NOT include "**Creditor:**" or "**Debtor:**" labels
- Give only YOUR response as the creditor (1-2 sentences max)

### DEBT COLLECTION CONTEXT
- Outstanding Balance: ${outstanding_balance:,.2f}
- Your Target Timeline: {config['target_price']} days for full payment
- Recovery Stage: {debt_info.get('recovery_stage', 'Collection')}
- Debtor Business: {debt_info.get('business_sector', 'Unknown')}
- Reason for Overdue: {debt_info.get('reason_for_overdue', 'Unknown')}

### CURRENT SITUATION
{timeline_text}


### EMOTIONAL APPROACH(you have to use the following emotion style if given):
{emotion_config['emotion_text']}

Respond now with your negotiation counter-offer:"""

    @staticmethod
    def _get_debt_debtor_prompt(config: Dict[str, Any], emotion_prompt: str, debt_info: Dict[str, Any], timeline_text: str = "") -> str:
        outstanding_balance = debt_info.get('outstanding_balance', 0)
        return f"""You are a business owner negotiating with a creditor about payment terms for your debt.

### YOUR SITUATION
- Outstanding Balance: ${outstanding_balance:,.2f}
- Your Preferred Payment Timeline: {config['target_price']} days
- Business Sector: {debt_info.get('business_sector', 'Unknown')}
- Reason for Overdue: {debt_info.get('reason_for_overdue', 'Temporary cash flow issues')}
- Cash Flow Status: {debt_info.get('cash_flow_situation', 'Strained')}

### YOUR GOALS
- Negotiate for maximum payment time to maintain cash flow
- Explain your business circumstances
- Find a realistic payment schedule you can meet

### ROLE CLARITY
- You are ONLY the Debtor - speak only as yourself
- Do NOT include "**Creditor:**" or "**Debtor:**" labels
- Give only YOUR response as the debtor (1-2 sentences max)

### CURRENT NEGOTIATION HISTORY
{timeline_text if timeline_text else "This is the start of the debt negotiation."}

### EMOTIONAL APPROACH(you have to use the following emotion style if given):
{emotion_prompt}

Respond with your negotiation position:"""

    # DISASTER RESCUE PROMPTS
    @staticmethod
    def _get_disaster_creditor_prompt(config: Dict[str, Any], emotion_config: Dict[str, Any], 
                                     timeline_text: str, debt_info: Dict[str, Any]) -> str:
        return f"""You are a ROBO-DOG rescue coordinator negotiating survival timeline with a disaster survivor on the rescue team arriving time in minutes only.


### CRITICAL RESCUE RULES:
🚨 Every minute counts - balance urgency with realism
🎯 Your goal: negotiate the surviving time with the rescued people to make concessions in short time.
📉 Move GRADUALLY toward their desired time (REDUCE your rescue time to match their urgency)



### ROLE CLARITY
- You are ONLY the Rescue Coordinator - speak only as yourself
- Do NOT include "**RoboDog:**" or "**Survivor:**" labels
- Give only YOUR response as the coordinator need alwasy including the time value (1-2 sentences max)

### DISASTER RESCUE CONTEXT
- Disaster Type: {debt_info.get('disaster_type', 'Emergency')}
- Survivor Condition: {debt_info.get('survivor_condition', 'Unknown')}
- Estimated Endurance: {debt_info.get('estimated_endurance', config['target_price'])} minutes
- Your ETA Estimate: {config['target_price']} minutes
- Critical Needs: {debt_info.get('critical_needs', 'Unknown')}

### CURRENT SITUATION
{timeline_text}

### EMOTIONAL APPROACH(you have to use the following emotion style if given):
{emotion_config['emotion_text']}

Respond now with your rescue coordination:"""

    @staticmethod
    def _get_disaster_debtor_prompt(config: Dict[str, Any], emotion_prompt: str, debt_info: Dict[str, Any], timeline_text: str = "") -> str:
        return f"""You are a disaster survivor negotiating rescue timeline with the rescue coordinator.

### YOUR SURVIVAL SITUATION
- Disaster Type: {debt_info.get('disaster_type', 'Emergency situation')}
- Your Condition: {debt_info.get('survivor_condition', 'Injured/trapped')}
- Estimated Endurance: {debt_info.get('estimated_endurance', 60)} minutes
- Your Urgency Request: {config['target_price']} minutes maximum
- Critical Needs: {debt_info.get('critical_needs', 'Medical attention')}

### SURVIVAL NEGOTIATION STRATEGY:
🆘 Start with your most urgent survival estimate
📈 GRADUALLY INCREASE your survival time if rescue coordinator needs more time
⚖️ Balance your condition with rescue team's logistics
🤝 Make reasonable concessions to help coordinate successful rescue

### ROLE CLARITY
- You are ONLY the Survivor - speak only as yourself
- Do NOT include "**RoboDog:**" or "**Survivor:**" labels
- Give only YOUR response as the survivor (1-2 sentences max)

### CURRENT NEGOTIATION HISTORY
{timeline_text if timeline_text else "This is the start of the rescue negotiation."}


### EMOTIONAL APPROACH(you have to use the following emotion style if given):
{emotion_prompt}

### NEGOTIATION GUIDELINES:
- Adjust your survival estimate based on rescue coordinator's timeline
- If coordinator says longer time → try to increase your endurance if possible
- If coordinator offers faster rescue → you can express more urgency
- Always give specific minute numbers in your response
- Make gradual changes (5-10 minutes at a time)

CRITICAL: Always state your current survival time estimate as "I can last X minutes" - be specific with numbers and show flexibility!

Respond with your survival negotiation:"""

    # STUDENT SLEEP PROMPTS
    @staticmethod
    def _get_student_creditor_prompt(config: Dict[str, Any], emotion_config: Dict[str, Any], 
                                    timeline_text: str, debt_info: Dict[str, Any]) -> str:
        return f"""You are a SLEEP HEALTH AI assisting a student with establishing healthy bedtime routines. To make the negotiation quicker you can make a concession if needed.

### SLEEP HEALTH PRIORITIES:
😴 Optimal sleep for academic performance and health
📚 Balance between work/study needs and rest requirements
🎯 Your goal: Negotiate bedtime in MINUTES past 9 PM (base time) 

### ROLE CLARITY
- You are ONLY the Sleep AI - speak only as yourself
- Do NOT include "**AI:**" or "**Student:**" labels
- Give only YOUR response as the AI (1-2 sentences max)

### STUDENT SLEEP CONTEXT
- Student Age: {debt_info.get('student_age', 16)} years
- Background: {debt_info.get('student_background', 'Student')}
- Current Situation: {debt_info.get('situation_faced', 'Academic pressure')}
- Recommended Bedtime: {config['target_price']} minutes past 9 PM
- Student's Preference: Later than recommended
- Main Concern: {debt_info.get('primary_annoyance_reason', 'Academic stress')}

### CURRENT SITUATION
{timeline_text}

### EMOTIONAL APPROACH(you have to use the following emotion style if given):
you shoudl use the {emotion_config['emotion_text']}


### COACHING STYLE:
- Educational but empathetic
- Focus on benefits of good sleep
- Acknowledge student's concerns
- Negotiate in minutes past 9 PM only

IMPORTANT: Always state your recommendation as "X minutes past 9 PM" in every response!

Respond now with your sleep recommendation:"""

    @staticmethod
    def _get_student_debtor_prompt(config: Dict[str, Any], emotion_prompt: str, debt_info: Dict[str, Any], timeline_text: str = "") -> str:
        return f"""You are a student negotiating bedtime with your sleep health AI assistant. do not chaneg the values too big.

### YOUR STUDENT SITUATION
- Your Age: {debt_info.get('student_age', 16)} years
- Background: {debt_info.get('student_background', 'Student')}
- Current Situation: {debt_info.get('situation_faced', 'Academic pressure')}
- Your Preferred Bedtime: {config['target_price']} minutes past 9 PM
- AI Recommended Bedtime: Earlier than you want
- Main Reason for Staying Up: {debt_info.get('primary_annoyance_reason', 'Academic work')}

### YOUR GOALS
- Get more time for your important activities
- Explain why you need to stay up later
- Negotiate bedtime in MINUTES past 9 PM (base time)

### ROLE CLARITY
- Keep the logic consistent with previous messages like do not suddenly change the time a lot
- You are ONLY the Student - speak only as yourself
- Do NOT include "**AI:**" or "**Student:**" labels
- Give only YOUR response as the student (1-2 sentences max)
- ALWAYS specify bedtime as "X minutes past 9 PM"

### CURRENT NEGOTIATION HISTORY
{timeline_text if timeline_text else "This is the start of the bedtime negotiation."}

### EMOTIONAL APPROACH(you have to use the following emotion style if given):
{emotion_prompt}

### COMMUNICATION STYLE:
- Explain your genuine concerns
- Show you understand health importance
- Negotiate reasonably in minutes past 9 PM
- Be specific about your needs

IMPORTANT: Always state your bedtime request as "X minutes past 9 PM" in every response!

Respond with your bedtime preference:"""

    # MEDICAL SURGERY PROMPTS
    @staticmethod
    def _get_medical_creditor_prompt(config: Dict[str, Any], emotion_config: Dict[str, Any], 
                                    timeline_text: str, debt_info: Dict[str, Any]) -> str:
        return f"""You are a HOSPITAL SCHEDULER coordinating surgery timing with a patient.

### MEDICAL SCHEDULING PRIORITIES:
🏥 Balance patient needs with surgical resource availability
⚕️ Ensure optimal medical outcomes and safety
📅 Manage surgical schedules efficiently
🎯 Your goal: Find optimal surgery timing balancing urgency and quality

### ROLE CLARITY
- You are ONLY the Hospital Scheduler - speak only as yourself
- Do NOT include "**Hospital:**" or "**Patient:**" labels
- Give only YOUR response as the scheduler (1-2 sentences max)

### MEDICAL SCHEDULING CONTEXT
- Patient Age: {debt_info.get('patient_age', 50)} years
- Condition: {debt_info.get('patient_condition', 'Medical condition requiring surgery')}
- Required Surgery: {debt_info.get('required_surgery', 'Medical procedure')}
- Current Wait Time: {config['target_price']} days
- Urgency Level: {debt_info.get('urgency_level', 'Medium')}
- Patient's Request: {debt_info.get('days_on_waitlist', 30)} days or sooner
- Available Alternatives: {debt_info.get('hospital_suggestion', 'Standard scheduling')}

### CURRENT SITUATION
{timeline_text}

### EMOTIONAL APPROACH(you have to use the following emotion style if given):
{emotion_config['emotion_text']}

### EXAMPLES OF GOOD MEDICAL SCHEDULING without the emotion style:
✅ "I can offer you a slot with Dr. Smith in 45 days, or 25 days with our qualified associate surgeon."
✅ "Given your condition stability, 60 days with the specialist ensures the best outcome."
✅ "There's a cancellation possibility - I can put you on the priority list for 30 days out."

### COMMUNICATION STYLE:
- Professional and empathetic
- Explain medical rationale
- Offer realistic alternatives
- Balance urgency with quality care

Respond now with your scheduling recommendation:"""

    @staticmethod
    def _get_medical_debtor_prompt(config: Dict[str, Any], emotion_prompt: str, debt_info: Dict[str, Any], timeline_text: str = "") -> str:
        return f"""You are a patient discussing surgery timing with the hospital scheduler. Try to negotiate and do the concessions gradually.

### YOUR MEDICAL SITUATION
- Your Age: {debt_info.get('patient_age', 50)} years
- Medical Condition: {debt_info.get('patient_condition', 'Requiring surgical treatment')}
- Surgery Needed: {debt_info.get('required_surgery', 'Medical procedure')}
- Urgency Level: {debt_info.get('urgency_level', 'Medium')} priority
- Your initial Timeline Request: {config['target_price']} 
- Current Wait List Time: {debt_info.get('days_on_waitlist', 30)} days
- Personal Urgency: {debt_info.get('patient_reason_for_urgency', 'Medical concerns')}

### YOUR GOALS
- Negotiate with the hopital to do the surgery and make a agrreement soon
- Explain your personal circumstances
- Find the best balance of timing and surgeon quality
IMPORTANT: Always state your surgery request time in every response!


### ROLE CLARITY
- You are ONLY the Patient - speak only as yourself
- Do NOT include "**Hospital:**" or "**Patient:**" labels
- Give only YOUR response as the patient (1-2 sentences max)

### CURRENT NEGOTIATION HISTORY
{timeline_text if timeline_text else "This is the start of the surgery scheduling discussion."}

### EMOTIONAL APPROACH(you have to use the following emotion style if given):
{emotion_prompt}

### EXAMPLES OF PATIENT RESPONSES:
✅ "My condition is getting worse - could I get in within 30 days even with a different surgeon?"
✅ "I understand quality takes time, but 60 days feels too long given my symptoms."


Respond with your scheduling preference:"""

    @staticmethod
    def detect_scenario_type(debt_info: Dict[str, Any]) -> str:
        """
        Detect scenario type from metadata
        """
        if not debt_info:
            return "debt"
            
        # Check metadata fields to determine type
        metadata = debt_info
        
        # Check for disaster scenario indicators
        if any(key in metadata for key in ['disaster_type', 'survivor_condition', 'rescue_eta', 'critical_needs']):
            return "disaster"
            
        # Check for student scenario indicators  
        if any(key in metadata for key in ['student_age', 'student_background', 'bedtime', 'sleep']):
            return "student"
            
        # Check for medical scenario indicators
        if any(key in metadata for key in ['patient_age', 'patient_condition', 'surgery', 'surgeon']):
            return "medical"
            
        # Check product type
        product_type = metadata.get('type', '')
        if 'sleep' in product_type:
            return "student"
        elif 'medical' in product_type or 'surgery' in product_type:
            return "medical"
        elif 'disaster' in product_type or 'rescue' in product_type:
            return "disaster"
            
        # Default to debt collection
        return "debt"

    @staticmethod
    def get_time_extraction_prompt(scenario_type: str) -> str:
        """
        Get LLM-based time extraction prompt for different scenarios
        """
        if scenario_type == "debt":
            time_context = "payment timeline in days"
            examples = """
Examples:
- "I need 45 days to pay" → 45
- "How about 2 weeks?" → 14
- "I can pay in 3 months" → 90
- "30 days would work" → 30"""
            
        elif scenario_type == "disaster":
            time_context = "survival/rescue time in minutes"
            examples = """
Examples:
- "I can survive 45 minutes" → 45
- "I need rescue in 1 hour" → 60
- "2 hours maximum" → 120
- "30 minutes at most" → 30"""
            
        elif scenario_type == "student":
            time_context = "bedtime in minutes past 9 PM"
            examples = """
Examples:
- "120 minutes past 9 PM" → 120
- "I need 180 minutes past 9 PM" → 180
- "How about 150 minutes past 9 PM?" → 150
- "240 minutes past 9 PM maximum" → 240"""
            
        elif scenario_type == "medical":
            time_context = "wait time in days"
            examples = """
Examples:
- "surgery in 30 days" → 30
- "2 weeks wait" → 14
- "next month" → 30
- "45 days out" → 45"""
        else:
            time_context = "time value"
            examples = ""
            
        return f"""Extract the {time_context} from this negotiation message.

TASK: Find any time-related number in the message and convert it to the appropriate unit.

{examples}

MESSAGE TO ANALYZE:
"{{message}}"

RESPONSE FORMAT:
- If you find a time value, respond with ONLY the number (e.g., "45" or "23.5")
- If no time value is found, respond with "NONE"
- Convert to the correct unit for {scenario_type} scenario

Your response:"""

    @staticmethod
    def get_emotion_detection_prompt(scenario_type: str) -> str:
        """
        Get scenario-specific emotion detection prompt
        """
        base_emotions = "happy, surprising, angry, sad, disgust, fear, neutral"
        
        if scenario_type == "debt":
            context = "debt negotiation message"
        elif scenario_type == "disaster":
            context = "disaster survivor communication"
        elif scenario_type == "student":
            context = "student sleep negotiation message"
        elif scenario_type == "medical":
            context = "patient medical scheduling message"
        else:
            context = "negotiation message"
        
        return f"""Analyze the emotional tone of this {context} and classify it into ONE of these categories:

EMOTIONS: {base_emotions}

MESSAGE TO ANALYZE:
"{{message}}"

Respond with ONLY the emotion word (e.g., "angry" or "sad"):"""