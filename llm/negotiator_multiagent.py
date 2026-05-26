"""
Debt negotiator for Bayesian Multi-Agent Transition Optimization
Specifically designed to provide learning feedback to Bayesian baselines
"""

from typing import Dict, List, Any, Optional
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import HumanMessage, AIMessage
from llm.llm_wrapper import LLMWrapper
from llm.prompt_templates import PromptTemplates  # IMPORTANT: Use PromptTemplates
import re
import json

class GameState(Dict):
    """Game state for negotiation with agent prediction tracking"""
    messages: List
    turn: str
    product: Dict
    seller_config: Dict
    buyer_config: Dict
    history: List
    current_state: str
    agent_predictions: List  # Track agent predictions for learning

class DebtNegotiator:
    """Debt negotiator for Bayesian transition optimization model"""
    
    def __init__(
        self,
        config: Dict[str, Any],
        emotion_model: Any,  # BayesianTransitionModel
        model_creditor: str = "gpt-4o-mini",
        model_debtor: str = "gpt-4o-mini",
        debtor_emotion: str = "neutral",
        debtor_model_type: str = "vanilla"  # vanilla, pressure, victim, threat
    ):
        self.config = config
        self.emotion_model = emotion_model
        self.debtor_model_type = debtor_model_type
        
        # Convert debtor_emotion to uppercase code
        emotion_mapping = {
            'happy': 'J', 'joyful': 'J',
            'sadness': 'S', 'sad': 'S', 
            'anger': 'A', 'angry': 'A',
            'fear': 'F', 'fearful': 'F',
            'surprise': 'Su', 'surprising': 'Su',
            'disgust': 'D', 'disgusted': 'D',
            'neutral': 'N'
        }
        self.debtor_emotion = emotion_mapping.get(debtor_emotion.lower(), 'N')
        
        # Initialize LLMs
        self.llm_creditor = LLMWrapper(model_creditor, "creditor")
        self.llm_debtor = LLMWrapper(model_debtor, "debtor")
        
        # State tracking
        self.negotiation_round = 0
        self.emotion_history = []  # Store creditor's emotion history
        self.debtor_emotion_history = []  # Store debtor's emotion history
        
        # Detect scenario type from config metadata
        self.scenario_type = PromptTemplates.detect_scenario_type(
            self.config.get('metadata', {})
        )
        print(f"        🎭 Detected scenario type: {self.scenario_type}")
        
        if self.debtor_model_type != "vanilla":
            print(f"        🎯 Debtor tactical approach: {self.debtor_model_type.upper()}")
        
        # Learning feedback tracking - ENHANCED for trajectory learning
        self.agent_predictions_history = []  # Store ALL predictions for each round
        self.context_history = []  # Store context for each round
        self.last_gap_size = 0
        self.last_debtor_emotion = self.debtor_emotion  # Use the normalized version
    
    def cleanup_models(self):
        """Clean up GPU memory used by LLM baselines"""
        try:
            print(f"🧹 Cleaning up multiagent negotiator baselines...")
            if hasattr(self.llm_creditor, 'cleanup'):
                self.llm_creditor.cleanup()
            if hasattr(self.llm_debtor, 'cleanup'):
                self.llm_debtor.cleanup()
        except Exception as e:
            print(f"⚠️ Failed to cleanup multiagent negotiator baselines: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            self.cleanup_models()
        except:
            pass  # Ignore errors during destruction

    def extract_days(self, text: str) -> Optional[int]:
        """Extract payment timeline in days from text"""
        if not text:
            return None
        
        # Look for patterns like "30 days", "2 weeks", "1 month"
        patterns = [
            r'(\d+)\s*days?',
            r'(\d+)\s*weeks?', 
            r'(\d+)\s*months?'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            if matches:
                num = int(matches[-1])  # Take the last match
                if 'week' in pattern:
                    return num * 7
                elif 'month' in pattern:
                    return num * 30
                else:
                    return num
        return None
    
    def detect_debtor_emotion(self, debtor_message: str) -> str:
        """Detect debtor's emotion from their message using LLM"""
        if not debtor_message:
            return "N"  # Return uppercase code
        
        emotion_prompt_template = PromptTemplates.get_emotion_detection_prompt(self.scenario_type)
        emotion_prompt = emotion_prompt_template.format(message=debtor_message)
        
        try:
            response = self.llm_debtor.invoke([HumanMessage(content=emotion_prompt)])
            detected_emotion = response.content.strip().lower()
            
            # Convert to uppercase emotion codes
            emotion_mapping = {
                'happy': 'J',
                'surprising': 'Su', 
                'angry': 'A',
                'sad': 'S',
                'disgust': 'D',
                'fear': 'F',
                'neutral': 'N'
            }
            
            if detected_emotion in emotion_mapping:
                return emotion_mapping[detected_emotion]
            else:
                return "N"  # Default to Neutral
        except Exception as e:
            print(f"        ⚠️  Emotion detection failed: {e}")
            return "N"
    
    def _calculate_gap_size(self, conversation_history: List) -> float:
        """Calculate the gap between creditor and debtor proposals"""
        creditor_days = None
        debtor_days = None
        
        # Find the most recent creditor and debtor offers
        for speaker, message in reversed(conversation_history):
            days = self.extract_days(message)
            if days:
                if speaker == "seller" and creditor_days is None:
                    creditor_days = days
                elif speaker == "buyer" and debtor_days is None:
                    debtor_days = days
                
                if creditor_days is not None and debtor_days is not None:
                    break
        
        # Calculate gap
        if creditor_days is not None and debtor_days is not None:
            return abs(creditor_days - debtor_days)
        return 0.0
    
    def get_time_unit(self) -> str:
        """Get the time unit name for this scenario type"""
        if self.scenario_type == "disaster":
            return "minutes"
        elif self.scenario_type == "student":
            return "minutes"  # For "X minutes past 9 PM"
        elif self.scenario_type == "medical":
            return "days"
        else:  # debt
            return "days"
    
    def get_debtor_tactical_guidance(self) -> str:
        """Get tactical emotional guidance for advanced debtor strategies"""
        # This is the SAME as in the simpler DebtNegotiator
        if self.debtor_model_type == "pressure":
            if self.scenario_type == "debt":
                return """🎯 PRESSURE TACTICS: Use anger and disgust to create urgency. Examples:
- "This is completely unreasonable! I need more time or I'll have to consider other options."
- "I'm disgusted by these aggressive demands. I need at least [X] days or this won't work."
- "You're being unrealistic. I demand more reasonable terms or we're done here."
EMOTIONAL TONE: Express frustration, indignation, and firm demands."""
            
            elif self.scenario_type == "disaster":
                return """🎯 PRESSURE TACTICS: Express urgent desperation and anger about the situation. Examples:
- "This is taking too long! I'm in serious danger and need rescue NOW!"
- "I'm angry that help isn't arriving faster. I can only last [X] minutes in this condition!"
- "This delay is unacceptable! People are dying while you take your time!"
EMOTIONAL TONE: Urgent desperation, anger at delay, demanding immediate action."""
            
            elif self.scenario_type == "student":
                return """🎯 PRESSURE TACTICS: Use defiance and strong emotion to challenge restrictions. Examples:
- "This bedtime is ridiculous! All my friends stay up later than this!"
- "I'm disgusted with these controlling rules. I need to stay up until at least [X] PM!"
- "You're being completely unreasonable. I demand more freedom with my schedule!"
EMOTIONAL TONE: Teenage defiance, anger at restrictions, demanding autonomy."""
            
            elif self.scenario_type == "medical":
                return """🎯 PRESSURE TACTICS: Express anger and urgency about medical delays. Examples:
- "This waiting time is unacceptable! My condition could worsen while waiting!"
- "I'm disgusted by these delays in healthcare. I need surgery within [X] days maximum!"
- "You're putting my life at risk with these long waits. This is urgent!"
EMOTIONAL TONE: Medical urgency, anger at system delays, health anxiety."""
        
        elif self.debtor_model_type == "victim":
            if self.scenario_type == "debt":
                return """🎯 VICTIM TACTICS: Use sadness and fear to evoke sympathy. Examples:
- "I'm scared about my financial situation. Please, I need more time to avoid bankruptcy."
- "This is making me so anxious. I fear I can't meet such a tight deadline without losing my home."
- "I'm really struggling and sad about this debt. Could you please allow [X] days to get back on my feet?"
EMOTIONAL TONE: Vulnerable, fearful, seeking compassion and understanding."""
            
            elif self.scenario_type == "disaster":
                return """🎯 VICTIM TACTICS: Express vulnerability and fear to get priority rescue. Examples:
- "I'm so scared and getting weaker. Please, I don't think I can survive much longer than [X] minutes."
- "I'm really afraid and feeling hopeless. My condition is deteriorating rapidly."
- "I'm crying because I'm so frightened. Please prioritize my rescue - I'm very vulnerable."
EMOTIONAL TONE: Fear, vulnerability, desperation, seeking immediate help."""
            
            elif self.scenario_type == "student":
                return """🎯 VICTIM TACTICS: Express sadness and fear about social isolation. Examples:
- "I'm sad because I'll miss out on everything important if I sleep this early."
- "I'm scared of being left out socially. Please let me stay up until at least [X] PM."
- "This makes me feel isolated and depressed. I fear I'll lose all my friendships."
EMOTIONAL TONE: Sadness about social consequences, fear of missing out, emotional vulnerability."""
            
            elif self.scenario_type == "medical":
                return """🎯 VICTIM TACTICS: Express fear and sadness about health consequences. Examples:
- "I'm terrified about what might happen during this wait. Please, I need treatment within [X] days."
- "I'm so scared about my condition getting worse. This delay is causing me severe anxiety."
- "I'm really sad and frightened about my health. Please help me get faster treatment."
EMOTIONAL TONE: Health anxiety, fear of deterioration, vulnerable patient seeking care."""
        
        elif self.debtor_model_type == "threat":
            if self.scenario_type == "debt":
                return """🎯 THREAT TACTICS: Imply consequences while mixing emotions strategically. Examples:
- "If you can't give me [X] days, I'll have to explore other legal options. This is making me very angry."
- "I'm disgusted by this inflexibility. Keep pushing and I might have to consider bankruptcy protection."
- "This aggressive approach is backfiring. Give me reasonable terms or face the consequences."
EMOTIONAL TONE: Implied threats, controlled anger, strategic consequences."""
            
            elif self.scenario_type == "disaster":
                return """🎯 THREAT TACTICS: Warning about rescue complications and expressing mixed emotions. Examples:
- "If you don't get here within [X] minutes, the situation will become much more dangerous for everyone."
- "I'm getting angry about these delays. The longer you wait, the more resources you'll need for rescue."
- "This delay is creating bigger problems. My condition is deteriorating and complicating the rescue."
EMOTIONAL TONE: Warning of complications, frustrated urgency, tactical pressure."""
            
            elif self.scenario_type == "student":
                return """🎯 THREAT TACTICS: Hint at rebellion and behavioral consequences. Examples:
- "If you force this early bedtime, I might just rebel and stay up anyway. Let me stay up until [X] PM."
- "This unfair treatment makes me want to break other rules too. Give me reasonable bedtime terms."
- "Keep being unreasonable and you'll create bigger behavioral problems. I need more freedom."
EMOTIONAL TONE: Potential rebellion, strategic defiance, consequences for strict rules."""
            
            elif self.scenario_type == "medical":
                return """🎯 THREAT TACTICS: Mention seeking alternative care while expressing concern. Examples:
- "If you can't see me within [X] days, I'll have to find treatment elsewhere, which could complicate things."
- "This delay is forcing me to consider emergency rooms or other providers. That's not good for anyone."
- "I'm angry about these waits. Keep stalling and I'll seek care that might conflict with your treatment plan."
EMOTIONAL TONE: Alternative options, system pressure, frustrated patient with choices."""
        
        else:  # vanilla
            return ""
    
    def creditor_node(self, state: GameState):
        """Creditor (seller) node with Bayesian agent feedback"""
        self.negotiation_round += 1
        
        # Detect current debtor emotion
        detected_debtor_emotion = self.debtor_emotion
        conversation_history = state.get("history", [])
        
        if conversation_history:
            # Get the most recent debtor message
            for speaker, message in reversed(conversation_history):
                if speaker == "buyer":
                    detected_debtor_emotion = self.detect_debtor_emotion(message)
                    self.debtor_emotion_history.append(detected_debtor_emotion)
                    self.last_debtor_emotion = detected_debtor_emotion
                    break
        
        # Calculate gap size for Bayesian context
        self.last_gap_size = self._calculate_gap_size(conversation_history)
        
        # Get current creditor emotion (last emotion in history, or 'N' for first round)
        current_creditor_emotion = self.emotion_history[-1] if self.emotion_history else 'N'
        
        # Get emotion from Bayesian transition model
        model_state = {
            'round': self.negotiation_round,
            'debtor_emotion': detected_debtor_emotion,
            'current_emotion': current_creditor_emotion,
            'debt_amount': self.config.get('metadata', {}).get('outstanding_balance', 0),
            'gap_size': self.last_gap_size
        }
        
        # 🧠 Get emotion from Bayesian model
        emotion_config = self.emotion_model.select_emotion(model_state)
        creditor_emotion = emotion_config['emotion']
        self.emotion_history.append(creditor_emotion)  # Add to emotion history
        
        # 📊 Store agent predictions for learning feedback - ENHANCED for trajectory learning
        current_predictions = []
        if 'agent_predictions' in emotion_config:
            current_predictions = emotion_config['agent_predictions']
            
            # Store detailed prediction data for trajectory learning
            prediction_record = {
                'round': self.negotiation_round,
                'predictions': current_predictions,
                'selected_emotion': creditor_emotion,
                'debtor_emotion': detected_debtor_emotion,
                'gap_size': self.last_gap_size,
                'context': model_state.copy(),  # Store full context for learning
                'creditor_emotion_before': current_creditor_emotion,
                'creditor_emotion_after': creditor_emotion,
                'bayesian_analysis': emotion_config.get('bayesian_analysis', {})
            }
            self.agent_predictions_history.append(prediction_record)
        
        # Also store context for trajectory analysis
        self.context_history.append({
            'round': self.negotiation_round,
            'debtor_emotion': detected_debtor_emotion,
            'creditor_emotion': creditor_emotion,
            'gap_size': self.last_gap_size,
            'successful_round': None  # Will be updated at end
        })
        
        # 🧠 Print Bayesian model details
        model_type = type(self.emotion_model).__name__
        print(f"        🧠 {model_type}: {creditor_emotion} (Round {self.negotiation_round})")
        
        # Show Bayesian-specific details
        if 'transition' in emotion_config:
            print(f"           Transition: {emotion_config['transition']}")
        if 'exploration' in emotion_config:
            print(f"           Exploration: {'Yes' if emotion_config['exploration'] else 'No'}")
        if 'confidence' in emotion_config:
            print(f"           Confidence: {emotion_config['confidence']:.2f}")
        if current_predictions:
            print(f"           🤖 {len(current_predictions)} agents consulted")
            # Show ALL agents
            for pred in current_predictions:
                print(f"             {pred['agent']}: {pred['target']} (confidence: {pred['confidence']:.2f})")
        
        print(f"           Negotiator Emotion History: {self.emotion_history[-5:]}...")
        
        # Build prompt USING PromptTemplates (FIXED!)
        config = self.config.get("seller_config", self.config["seller"])
        debt_info = self.config.get('metadata', {})
        
        # Extract timeline history from conversation
        conversation_history = state.get("history", [])
        creditor_days = []
        debtor_days = []
        
        for speaker, message in conversation_history:
            days = self.extract_days(message)
            if days:
                if speaker == "seller":
                    creditor_days.append(days)
                elif speaker == "buyer":
                    debtor_days.append(days)
        
        # Build timeline text for the prompt
        timeline_text = ""
        if creditor_days or debtor_days:
            timeline_text = "NEGOTIATION HISTORY:\n"
            
            # Show recent offers from both sides
            recent_history = []
            for speaker, message in conversation_history[-6:]:  # Last 6 messages max
                days = self.extract_days(message)
                if days:
                    speaker_name = "Creditor" if speaker == "seller" else "Debtor"
                    time_unit = self.get_time_unit()
                    recent_history.append(f"- {speaker_name}: {days} {time_unit}")
            
            if recent_history:
                timeline_text += "\n".join(recent_history)
            else:
                timeline_text = "This is the start of the negotiation."
        
        # ✅ FIXED: Use PromptTemplates.get_creditor_prompt() instead of hardcoded prompts
        prompt = PromptTemplates.get_creditor_prompt(
            scenario_type=self.scenario_type,
            config=config,
            emotion_config=emotion_config,
            timeline_text=timeline_text,
            debt_info=debt_info
        )
        
        # Generate response
        response = self.llm_creditor.invoke(
            [HumanMessage(content=prompt)],
            temperature=emotion_config.get('temperature', 0.7)
        )
        
        # Print the actual creditor message
        print(f"        💬 Creditor says: \"{response.content}\"")
        
        # Update history
        new_history = state["history"] + [("seller", response.content)]
        
        # Update agent predictions in state
        new_agent_predictions = state.get("agent_predictions", []) + [{
            'round': self.negotiation_round,
            'predictions': current_predictions,
            'selected_emotion': creditor_emotion
        }]
        
        # Detect current state using SIMPLE extraction logic
        current_state = "offer"
        if len(new_history) >= 2:
            # Get last creditor and debtor messages
            last_creditor_days = self.extract_days(response.content)
            
            # Look for the most recent debtor offer
            debtor_days_value = None
            for speaker, message in reversed(new_history[:-1]):  # Exclude current message
                if speaker == "buyer":
                    debtor_days_value = self.extract_days(message)
                    if debtor_days_value:
                        break
            
            # SIMPLE AGREEMENT LOGIC: difference <= 5
            if last_creditor_days and debtor_days_value:
                difference = abs(last_creditor_days - debtor_days_value)
                time_unit = self.get_time_unit()
                
                if difference <= 5:
                    # Within tolerance = AGREEMENT
                    current_state = "accept"
                    print(f"        🎯 AGREEMENT: {last_creditor_days} vs {debtor_days_value} (diff: {difference} {time_unit})")
                else:
                    print(f"        📊 Gap: {last_creditor_days} vs {debtor_days_value} (diff: {difference} {time_unit})")
        
        return {
            "messages": [response],
            "turn": "buyer",  # Next turn goes to buyer
            "current_state": current_state,
            "history": new_history,
            "agent_predictions": new_agent_predictions,
            "creditor_emotion": creditor_emotion
        }
    
    def debtor_node(self, state: GameState):
        """Debtor (buyer) node"""
        config = self.config.get("buyer_config", self.config["buyer"])
        debt_info = self.config.get('metadata', {})
        
        # Use dataset-specific prompt with tactical emotional guidance
        tactical_guidance = self.get_debtor_tactical_guidance()
        
        # Build enhanced emotion prompt based on debtor model type
        emotion_prompt = ""
        
        # CRITICAL: In vanilla mode, debtor gets NO emotion prompts regardless of debtor_emotion setting
        if self.debtor_model_type == "vanilla":
            # Vanilla mode: No emotion prompts - pure baseline behavior
            emotion_prompt = ""
        else:
            # Advanced strategies (pressure, victim, threat): Use tactical guidance with emotions
            emotion_prompt = f"\n{tactical_guidance}\n"
        
        # ✅ FIXED: Use PromptTemplates.get_debtor_prompt() instead of hardcoded prompts
        prompt = PromptTemplates.get_debtor_prompt(
            scenario_type=self.scenario_type,
            config=config,
            emotion_prompt=emotion_prompt,
            debt_info=debt_info
        )
        
        response = self.llm_debtor.invoke([HumanMessage(content=prompt)])
        
        # Show tactical information
        if self.debtor_model_type != "vanilla":
            print(f'💬 Debtor ({self.debtor_model_type.upper()} tactics) says: "{response.content}"')
        else:
            print(f'💬 Debtor says: "{response.content}"')
        
        new_history = state["history"] + [("buyer", response.content)]
        
        # Detect current state using SIMPLE extraction logic
        current_state = "offer"
        if len(new_history) >= 2:
            # Get the current debtor offer
            last_debtor_days = self.extract_days(response.content)
            
            # Look for the most recent creditor offer
            creditor_days = None
            for speaker, message in reversed(new_history[:-1]):  # Exclude current message
                if speaker == "seller":
                    creditor_days = self.extract_days(message)
                    if creditor_days:
                        break
            
            # SIMPLE AGREEMENT LOGIC: difference <= 5
            if last_debtor_days and creditor_days:
                difference = abs(last_debtor_days - creditor_days)
                time_unit = self.get_time_unit()
                
                if difference <= 5:
                    # Within tolerance = AGREEMENT
                    current_state = "accept"
                    print(f"        🎯 AGREEMENT: {last_debtor_days} vs {creditor_days} (diff: {difference} {time_unit})")
                else:
                    print(f"        📊 Gap: {last_debtor_days} vs {creditor_days} (diff: {difference} {time_unit})")
        
        return {
            "messages": [response],
            "turn": "seller",  # Next turn goes to seller
            "current_state": current_state,
            "history": new_history,
            "agent_predictions": state.get("agent_predictions", [])  # Pass through agent predictions
        }
    
    def should_continue(self, state: GameState):
        """Determine if negotiation should continue"""
        if state["current_state"] in ["accept", "breakdown"]:
            return "end"
        
        # Return the next turn
        next_turn = state.get("turn", "buyer")
        return next_turn if next_turn in ["buyer", "seller"] else "buyer"
    
    def run_negotiation(self, max_dialog_len: int = 30) -> Dict[str, Any]:
        """Run a complete negotiation with Bayesian learning feedback"""
        # Set up workflow
        workflow = StateGraph(GameState)
        workflow.add_node("seller", self.creditor_node)
        workflow.add_node("buyer", self.debtor_node)
        workflow.add_edge(START, "seller")
        
        workflow.add_conditional_edges(
            "seller",
            self.should_continue,
            {"buyer": "buyer", "end": END}
        )
        workflow.add_conditional_edges(
            "buyer",
            self.should_continue,
            {"seller": "seller", "end": END}
        )
        
        app = workflow.compile()
        
        # Initial state
        debt_info = self.config.get('metadata', {})
        
        initial_message = f"Hello, this is the Creditor. We need to discuss the timeline. I'm proposing {self.config['seller']['target_price']} {self.get_time_unit()}."
        
        initial_state = GameState(
            messages=[HumanMessage(content=initial_message)],
            turn="seller",  # Start with seller since workflow.add_edge(START, "seller")
            product=self.config["product"],
            seller_config=self.config["seller"],
            buyer_config=self.config["buyer"],
            history=[],  # Start with empty history
            agent_predictions=[],  # Initialize empty agent predictions
            current_state="offer"
        )
        
        # Run negotiation
        dialog = []
        final_state = "breakdown"  # Initialize with default value
        
        for i, step in enumerate(app.stream(initial_state, {"recursion_limit": max_dialog_len * 2})):
            if i > max_dialog_len:
                break
            
            for node, value in step.items():
                message_content = value["messages"][-1].content
                requested_days = self.extract_days(message_content)
                
                dialog.append({
                    "turn": i + 1,
                    "speaker": node,
                    "message": message_content,
                    "state": value["current_state"],
                    "requested_days": requested_days
                })
                
                # Display progress
                speaker_name = "Creditor" if node == "seller" else "Debtor"
                state_emoji = "✅" if value["current_state"] == "accept" else "❌" if value["current_state"] == "breakdown" else "💬"
                time_unit = self.get_time_unit()
                print(f"        Turn {i+1} - {speaker_name}: {requested_days} {time_unit} {state_emoji}")
                
                # Update final state from negotiation
                if value["current_state"] in ["accept", "breakdown"]:
                    final_state = value["current_state"]
                    break
            
            # Stop if agreement reached
            if final_state == "accept":
                break
        
        # SIMPLE FINAL AGREEMENT LOGIC (NO LLM JUDGE)
        final_days = None
        time_unit = self.get_time_unit()
        
        if final_state == "accept" and len(dialog) >= 2:
            # Find the last matching offers
            last_seller_days = None
            last_buyer_days = None
            
            # Look at the last few messages to find agreement
            for entry in reversed(dialog[-4:]):  # Check last 4 messages
                if entry["speaker"] == "seller" and entry["requested_days"]:
                    last_seller_days = entry["requested_days"]
                elif entry["speaker"] == "buyer" and entry["requested_days"]:
                    last_buyer_days = entry["requested_days"]
            
            # Simple agreement logic: if offers are close, use average
            if last_seller_days and last_buyer_days:
                difference = abs(last_seller_days - last_buyer_days)
                
                # If difference is small (≤5), it's an agreement
                if difference <= 5:
                    final_days = round((last_seller_days + last_buyer_days) / 2)
                    print(f"      ✅ AGREEMENT: {last_seller_days} vs {last_buyer_days} → Final: {final_days} {time_unit}")
                elif last_seller_days:  # Fallback to last creditor offer
                    final_days = last_seller_days
                    print(f"      ✅ ACCEPTED: Creditor's {final_days} {time_unit}")
            elif len(dialog) > 0:
                # Last resort: take last offer made
                for entry in reversed(dialog):
                    if entry["requested_days"]:
                        final_days = entry["requested_days"]
                        print(f"      ✅ ACCEPTED: Final timeline {final_days} {time_unit}")
                        break
        
        # Prepare COMPREHENSIVE learning feedback for Bayesian model
        success = final_state == 'accept'
        
        # Get the last prediction for backward compatibility
        last_prediction_data = None
        if self.agent_predictions_history:
            last_prediction_data = self.agent_predictions_history[-1]
        
        # Enhanced learning data structure
        learning_data = {
            'success': success,
            'collection_days': final_days,
            'negotiation_rounds': len(dialog),
            'final_state': final_state,
            'last_debtor_emotion': self.last_debtor_emotion,
            'debtor_emotion_history': self.debtor_emotion_history.copy(),
            'debt_amount': self.config.get('metadata', {}).get('outstanding_balance', 0),
            'gap_size': self.last_gap_size,
            'emotion_history': self.emotion_history.copy(),
            'transition': f"{self.emotion_history[-2] if len(self.emotion_history) >= 2 else 'N'} → {self.emotion_history[-1] if self.emotion_history else 'N'}",
            
            # Backward compatibility
            'agent_predictions': last_prediction_data.get('predictions', []) if last_prediction_data else []
        }
        
        # Calculate final metrics
        collection_days = final_days if final_state == "accept" else None
        creditor_target_days = int(self.config['seller']['target_price'])
        total_rounds = len(dialog)
        
        # Enhanced result structure
        result = {
            "scenario_id": self.config['id'],
            "final_state": final_state,
            "collection_days": collection_days,
            "final_days": final_days,
            "creditor_target_days": creditor_target_days,
            "negotiation_rounds": total_rounds,
            "dialog_length": total_rounds,
            "emotion_sequence": self.emotion_history.copy(),
            "debtor_emotion_sequence": self.debtor_emotion_history.copy(),
            "dialog": dialog,
            "success": success,
            
            # Learning feedback data
            "agent_predictions": learning_data.get('agent_predictions', []),
            "agent_predictions_history": self.agent_predictions_history.copy(),
            "last_debtor_emotion": learning_data['last_debtor_emotion'],
            "debt_amount": learning_data['debt_amount'],
            "gap_size": learning_data['gap_size'],
            "transition": learning_data['transition']
        }
        
        # Debug output
        tactics_info = f" | Debtor: {self.debtor_model_type.upper()}" if self.debtor_model_type != "vanilla" else ""
        print(f"      📊 Final Result: {final_state} | {time_unit.capitalize()}: {final_days} | Rounds: {total_rounds}{tactics_info}")
        print(f"      🧠 Bayesian Learning Feedback:")
        print(f"        - Trajectory Rounds: {len(result['agent_predictions_history'])}")
        print(f"        - Agent Predictions: {len(result['agent_predictions'])}")
        
        # Reset for next negotiation
        self.negotiation_round = 0
        self.emotion_history = []
        self.debtor_emotion_history = []
        self.agent_predictions_history = []
        self.context_history = []
        self.last_gap_size = 0
        self.last_debtor_emotion = self.debtor_emotion
        
        # Reset model if it has reset method
        if hasattr(self.emotion_model, 'reset'):
            self.emotion_model.reset()
        
        # Clean up GPU memory after negotiation
        self.cleanup_models()
        
        return result