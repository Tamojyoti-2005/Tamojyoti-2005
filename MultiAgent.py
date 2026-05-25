import json
import os
import datetime
from typing import Dict, List, Any
from abc import ABC, abstractmethod
from pathlib import Path
import sqlite3
import threading
import urllib.parse
import urllib.request
import importlib
from queue import Queue
from enum import Enum

# Third-party imports
st: Any = None
try:
    st = importlib.import_module("streamlit")
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

speech_to_text = None
try:
    speech_to_text = importlib.import_module("streamlit_mic_recorder").speech_to_text
    STREAMLIT_MIC_AVAILABLE = True
except Exception:
    STREAMLIT_MIC_AVAILABLE = False

try:
    import speech_recognition as sr
    SPEECH_AVAILABLE = True
except ImportError:
    SPEECH_AVAILABLE = False
    print("Warning: speech_recognition not installed. Install with: pip install SpeechRecognition pydub")

try:
    from langdetect import detect, detect_langs
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    print("Warning: langdetect not installed. Install with: pip install langdetect")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    LOCAL_AI_AVAILABLE = True
except Exception as e:
    LOCAL_AI_AVAILABLE = False
    LOCAL_AI_IMPORT_ERROR = str(e)
    print(
        "Warning: Local AI dependencies are not available. "
        "Install with: pip install transformers torch torchvision"
    )


class AgentRole(Enum):
    """Different roles for AI agents"""
    COORDINATOR = "Coordinator"
    ANALYZER = "Analyzer"
    RESEARCHER = "Researcher"
    VALIDATOR = "Validator"
    SYNTHESIZER = "Synthesizer"
    QA_AGENT = "QA Agent"


class ConversationHistory:
    """Manages conversation storage and retrieval"""
    
    def __init__(self, db_path: str = "conversation_history.db"):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Initialize database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    user_input TEXT,
                    agent_response TEXT,
                    agent_role TEXT,
                    language_detected TEXT,
                    session_id TEXT
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    summary TEXT
                )
            ''')
            conn.commit()
    
    def save_conversation(self, session_id: str, user_input: str, agent_response: str, 
                         agent_role: str, language_detected: str):
        """Save conversation to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO conversations (user_input, agent_response, agent_role, language_detected, session_id)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_input, agent_response, agent_role, language_detected, session_id))
            conn.commit()
    
    def get_session_history(self, session_id: str) -> List[Dict]:
        """Retrieve session history"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT user_input, agent_response, agent_role, language_detected, timestamp
                FROM conversations
                WHERE session_id = ?
                ORDER BY timestamp ASC
            ''', (session_id,))
            results = cursor.fetchall()
            return [
                {
                    'user_input': r[0],
                    'agent_response': r[1],
                    'agent_role': r[2],
                    'language': r[3],
                    'timestamp': r[4]
                }
                for r in results
            ]
    
    def get_all_sessions(self) -> List[str]:
        """Get all session IDs"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT DISTINCT session_id FROM conversations ORDER BY session_id DESC')
            return [r[0] for r in cursor.fetchall()]
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a specific session"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM conversations WHERE session_id = ?', (session_id,))
                conn.commit()
            return True
        except Exception as e:
            print(f"Error deleting session: {e}")
            return False
    
    def clear_all_conversations(self) -> bool:
        """Clear all conversations and sessions"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM conversations')
                cursor.execute('DELETE FROM sessions')
                conn.commit()
            return True
        except Exception as e:
            print(f"Error clearing conversations: {e}")
            return False
    
    def get_session_count(self) -> int:
        """Get total number of sessions"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(DISTINCT session_id) FROM conversations')
            return cursor.fetchone()[0]
    
    def get_total_conversations(self) -> int:
        """Get total number of conversations"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM conversations')
            return cursor.fetchone()[0]


class LanguageDetector:
    """Detects language of input text"""
    
    @staticmethod
    def detect_language(text: str) -> Dict[str, Any]:
        """Detect language and return details"""
        if not LANGDETECT_AVAILABLE:
            return {'language': 'unknown', 'confidence': 0.0}
        
        try:
            language = detect(text)
            probabilities = detect_langs(text)
            confidence = max([p.prob for p in probabilities])
            
            language_names = {
                'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
                'it': 'Italian', 'pt': 'Portuguese', 'ru': 'Russian',
                'zh-cn': 'Chinese (Simplified)', 'zh-tw': 'Chinese (Traditional)',
                'ja': 'Japanese', 'ko': 'Korean', 'ar': 'Arabic', 'hi': 'Hindi'
            }
            
            return {
                'language_code': language,
                'language_name': language_names.get(language, language),
                'confidence': confidence
            }
        except Exception as e:
            return {'language': 'unknown', 'confidence': 0.0, 'error': str(e)}


class SpeechRecognizer:
    """Handles speech recognition"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer() if SPEECH_AVAILABLE else None
        self.microphone = sr.Microphone() if SPEECH_AVAILABLE else None
    
    def listen_to_speech(self, timeout: int = 10, phrase_time_limit: int = None) -> Dict[str, Any]:
        """Listen to user speech and convert to text"""
        if not SPEECH_AVAILABLE or not self.recognizer:
            return {'status': 'error', 'message': 'Speech recognition not available'}
        
        try:
            with self.microphone as source:
                print("🎤 Listening... (speak now)")
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            
            try:
                text = self.recognizer.recognize_google(audio)
                return {'status': 'success', 'text': text}
            except sr.UnknownValueValue:
                return {'status': 'error', 'message': 'Could not understand audio'}
            except sr.RequestError as e:
                return {'status': 'error', 'message': f'API error: {e}'}
        except sr.RequestError as e:
            return {'status': 'error', 'message': f'Microphone error: {e}'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}


class SharedMemory:
    """Shared memory/context for all agents"""
    
    def __init__(self):
        self.memory: Dict[str, Any] = {}
        self.lock = threading.Lock()
    
    def set(self, key: str, value: Any):
        """Store value in shared memory"""
        with self.lock:
            self.memory[key] = value
    
    def get(self, key: str) -> Any:
        """Retrieve value from shared memory"""
        with self.lock:
            return self.memory.get(key)
    
    def get_all(self) -> Dict:
        """Get all memory contents"""
        with self.lock:
            return self.memory.copy()
    
    def update(self, data: Dict):
        """Update multiple values"""
        with self.lock:
            self.memory.update(data)
    
    def clear(self):
        """Clear all memory"""
        with self.lock:
            self.memory.clear()
    
    def clear_key(self, key: str):
        """Clear specific key from memory"""
        with self.lock:
            if key in self.memory:
                del self.memory[key]


class Agent(ABC):
    """Base Agent class"""
    
    def __init__(self, name: str, role: AgentRole, shared_memory: SharedMemory):
        self.name = name
        self.role = role
        self.shared_memory = shared_memory
        self.task_queue = Queue()
        self.is_running = False
    
    @abstractmethod
    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a task and return result"""
        pass
    
    def start(self):
        """Start agent thread"""
        self.is_running = True
        thread = threading.Thread(target=self._run)
        thread.daemon = True
        thread.start()
    
    def _run(self):
        """Agent main loop"""
        while self.is_running:
            try:
                task = self.task_queue.get(timeout=1)
                result = self.process_task(task)
                self.shared_memory.set(f"{self.name}_result", result)
            except:
                pass
    
    def stop(self):
        """Stop agent"""
        self.is_running = False
    
    def assign_task(self, task: Dict[str, Any]):
        """Assign a task to the agent"""
        self.task_queue.put(task)


class CoordinatorAgent(Agent):
    """Coordinates tasks between agents"""
    
    def __init__(self, shared_memory: SharedMemory):
        super().__init__("Coordinator", AgentRole.COORDINATOR, shared_memory)
    
    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process coordination task"""
        return {
            'status': 'processed',
            'coordinator_note': f'Task {task.get("id")} coordinated',
            'task_breakdown': [
                {'agent': 'Analyzer', 'task': 'Analyze input'},
                {'agent': 'Researcher', 'task': 'Research topic'},
                {'agent': 'Validator', 'task': 'Validate results'},
                {'agent': 'Synthesizer', 'task': 'Synthesize response'}
            ]
        }


class AnalyzerAgent(Agent):
    """Analyzes input data"""
    
    def __init__(self, shared_memory: SharedMemory):
        super().__init__("Analyzer", AgentRole.ANALYZER, shared_memory)
    
    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze task"""
        input_text = task.get('input', '')
        
        analysis = {
            'input_length': len(input_text),
            'word_count': len(input_text.split()),
            'language_analysis': LanguageDetector.detect_language(input_text),
            'key_elements': self._extract_key_elements(input_text)
        }
        
        return analysis
    
    def _extract_key_elements(self, text: str) -> List[str]:
        """Extract key elements from text"""
        words = text.split()
        return [word for word in words if len(word) > 4][:10]


class ResearcherAgent(Agent):
    """Researches and gathers information"""
    
    def __init__(self, shared_memory: SharedMemory):
        super().__init__("Researcher", AgentRole.RESEARCHER, shared_memory)
    
    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Research task"""
        topic = task.get('topic', '')
        
        research_findings = {
            'topic': topic,
            'sources': self._generate_mock_sources(topic),
            'findings': self._generate_mock_findings(topic),
            'confidence': 0.85
        }
        
        return research_findings
    
    def _generate_mock_sources(self, topic: str) -> List[str]:
        """Generate mock sources"""
        return [
            f"Source 1: Analysis of {topic}",
            f"Source 2: Research on {topic}",
            f"Source 3: Case study in {topic}"
        ]
    
    def _generate_mock_findings(self, topic: str) -> Dict:
        """Generate mock findings"""
        return {
            'key_findings': [f'Finding 1 about {topic}', f'Finding 2 about {topic}'],
            'statistics': {'data_points': 42, 'coverage': '87%'}
        }


class ValidatorAgent(Agent):
    """Validates and checks information"""
    
    def __init__(self, shared_memory: SharedMemory):
        super().__init__("Validator", AgentRole.VALIDATOR, shared_memory)
    
    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Validate task"""
        data_to_validate = task.get('data', {})
        
        validation_results = {
            'validated': True,
            'errors': self._check_errors(data_to_validate),
            'warnings': self._check_warnings(data_to_validate),
            'quality_score': self._calculate_quality(data_to_validate)
        }
        
        return validation_results
    
    def _check_errors(self, data: Dict) -> List[str]:
        """Check for errors"""
        errors = []
        if not data:
            errors.append('Empty data')
        return errors
    
    def _check_warnings(self, data: Dict) -> List[str]:
        """Check for warnings"""
        warnings = []
        if isinstance(data, dict) and len(data) < 3:
            warnings.append('Limited data points')
        return warnings
    
    def _calculate_quality(self, data: Dict) -> float:
        """Calculate data quality"""
        if not data:
            return 0.0
        return min(1.0, len(data) / 10)


class SynthesizerAgent(Agent):
    """Synthesizes information into final response"""
    
    def __init__(self, shared_memory: SharedMemory):
        super().__init__("Synthesizer", AgentRole.SYNTHESIZER, shared_memory)
    
    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize task"""
        components = task.get('components', {})
        
        synthesis = {
            'final_response': self._generate_response(components),
            'report': self._generate_report(components),
            'summary': self._generate_summary(components),
            'recommendations': self._generate_recommendations(components)
        }
        
        return synthesis
    
    def _generate_response(self, components: Dict) -> str:
        """Generate final response"""
        parts = [f"Analysis: {components.get('analysis', 'N/A')}"]
        parts.append(f"Research: {components.get('research', 'N/A')}")
        parts.append(f"Validation: {components.get('validation', 'N/A')}")
        return "\n".join(parts)
    
    def _generate_report(self, components: Dict) -> Dict:
        """Generate detailed report"""
        return {
            'title': 'Multi-Agent Analysis Report',
            'date': datetime.datetime.now().isoformat(),
            'components': components
        }
    
    def _generate_summary(self, components: Dict) -> str:
        """Generate summary"""
        return f"Processed {len(components)} components with multi-agent system"
    
    def _generate_recommendations(self, components: Dict) -> List[str]:
        """Generate recommendations"""
        return [
            "Review analysis results",
            "Validate findings with stakeholders",
            "Implement recommended actions"
        ]


class QAAgent(Agent):
    """Question Answering Agent using GPT-2 with a free online knowledge fallback"""
    
    def __init__(self, shared_memory: SharedMemory):
        super().__init__("QA Agent", AgentRole.QA_AGENT, shared_memory)
        self.model_name = "gpt2"
        self.tokenizer = None
        self.model = None
        self.model_load_error = None
        self.conversation_history = []
    
    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process QA task using GPT-2 without an API key"""
        question = task.get('question', '')
        context = task.get('context', '')
        
        try:
            self.conversation_history.append({
                "role": "user",
                "content": question
            })
            
            answer = self._generate_local_answer(question, context)
            
            self.conversation_history.append({
                "role": "assistant",
                "content": answer
            })
            
            model_used = self.model_name if self.model else 'Wikipedia online fallback'
            return {
                'question': question,
                'answer': answer,
                'model': model_used,
                'confidence': 0.65 if self.model else 0.35,
                'success': True,
                'requires_api_key': False
            }
        
        except Exception as e:
            fallback_answer = self._fallback_answer(question, context)
            return {
                'question': question,
                'answer': fallback_answer,
                'model': 'Wikipedia online fallback',
                'confidence': 0.35,
                'success': True
            }
    
    def _ensure_model_loaded(self) -> bool:
        """Load GPT-2 when the QA agent is first used, downloading it if needed."""
        if self.model and self.tokenizer:
            return True
        
        if not LOCAL_AI_AVAILABLE:
            self.model_load_error = "transformers or torch is not installed"
            return False
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            return True
        except Exception as e:
            self.model_load_error = str(e)
            self.tokenizer = None
            self.model = None
            return False
    
    def _generate_local_answer(self, question: str, context: str = '') -> str:
        """Generate an answer with GPT-2, downloading the model online if needed."""
        if not self._ensure_model_loaded():
            return self._fallback_answer(question, context)
        
        prompt_parts = [
            "Answer the question clearly and helpfully.",
        ]
        if context:
            prompt_parts.append(f"Context: {context}")
        prompt_parts.append(f"Question: {question}")
        prompt_parts.append("Answer:")
        prompt = "\n".join(prompt_parts)
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=768
        )
        output_ids = self.model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=140,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id
        )
        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        answer = generated_text[len(prompt):].strip() if generated_text.startswith(prompt) else generated_text.strip()
        
        if not answer:
            return self._fallback_answer(question, context)
        
        for marker in ("\nQuestion:", "\nAnswer:", "Question:", "Answer:"):
            marker_index = answer.find(marker)
            if marker_index > 0:
                answer = answer[:marker_index].strip()
        
        return answer.split("\n\n")[0].strip()
    
    def _fallback_answer(self, question: str, context: str = '') -> str:
        """Use a free online source when GPT-2 cannot be loaded."""
        question_clean = question.strip()
        context_clean = context.strip()
        online_answer = self._answer_from_wikipedia(question_clean)
        
        if online_answer:
            return online_answer
        
        if context_clean:
            return (
                "I could not reach the online answer source, but based on the provided "
                f"context, the key question is: {question_clean}"
            )
        
        return (
            "I could not reach the online answer source right now. Please check your "
            "internet connection and try again."
        )
    
    def _answer_from_wikipedia(self, question: str) -> str:
        """Fetch a short no-key online answer from Wikipedia."""
        topic = self._extract_topic(question)
        if not topic:
            return ""
        
        try:
            encoded_topic = urllib.parse.quote(topic.replace(" ", "_"))
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded_topic}"
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "MultiAgentSystem/1.0"}
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                data = json.loads(response.read().decode("utf-8"))
            
            extract = data.get("extract", "").strip()
            if not extract:
                return ""
            
            return extract
        except Exception:
            return ""
    
    def _extract_topic(self, question: str) -> str:
        """Convert simple questions into a searchable encyclopedia topic."""
        cleaned = question.strip().strip("?").lower()
        aliases = {
            "ai": "Artificial intelligence",
            "a.i": "Artificial intelligence",
            "a.i.": "Artificial intelligence"
        }
        
        prefixes = (
            "what is ", "what are ", "who is ", "who are ",
            "define ", "explain ", "tell me about "
        )
        
        for prefix in prefixes:
            if cleaned.startswith(prefix):
                topic = cleaned[len(prefix):].strip()
                return aliases.get(topic, topic.title())
        
        return aliases.get(cleaned, cleaned.title())
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []


class MultiAgentSystem:
    """Main multi-agent system orchestrator"""
    
    def __init__(self, session_id: str = None):
        self.session_id = session_id or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.shared_memory = SharedMemory()
        self.conversation_history = ConversationHistory()
        self.speech_recognizer = SpeechRecognizer() if SPEECH_AVAILABLE else None
        self.language_detector = LanguageDetector()
        
        # Initialize agents
        self.agents = {
            'coordinator': CoordinatorAgent(self.shared_memory),
            'analyzer': AnalyzerAgent(self.shared_memory),
            'researcher': ResearcherAgent(self.shared_memory),
            'validator': ValidatorAgent(self.shared_memory),
            'synthesizer': SynthesizerAgent(self.shared_memory),
            'qa_agent': QAAgent(self.shared_memory)
        }
        
        self.start_agents()
    
    def start_agents(self):
        """Start all agents"""
        for agent in self.agents.values():
            agent.start()
    
    def stop_agents(self):
        """Stop all agents"""
        for agent in self.agents.values():
            agent.stop()
    
    def process_input(self, user_input: str = None, use_speech: bool = False, is_question: bool = False) -> Dict[str, Any]:
        """Process user input through the system"""
        
        if use_speech:
            if not SPEECH_AVAILABLE:
                return {'error': 'Speech recognition not available'}
            
            speech_result = self.speech_recognizer.listen_to_speech()
            if speech_result['status'] != 'success':
                return speech_result
            user_input = speech_result['text']
        
        if not user_input:
            return {'error': 'No input provided'}
        
        print(f"\n[INPUT] {user_input}")
        
        language_info = self.language_detector.detect_language(user_input)
        print(f"[LANGUAGE] {language_info.get('language_name', 'Unknown')}")
        
        # Text and speech inputs automatically use GPT-2 when they look like questions.
        if is_question or self._is_question(user_input):
            return self._process_question(user_input, language_info)
        else:
            return self._process_standard(user_input, language_info)
    
    def _is_question(self, text: str) -> bool:
        """Detect common text and speech questions without needing a separate menu option."""
        cleaned_text = text.strip().lower()
        if cleaned_text.endswith('?'):
            return True
        
        question_starters = (
            'what', 'who', 'where', 'when', 'why', 'how',
            'can', 'could', 'should', 'would', 'will',
            'is', 'are', 'am', 'was', 'were',
            'do', 'does', 'did',
            'tell me', 'explain', 'define'
        )
        
        return any(
            cleaned_text == starter or cleaned_text.startswith(f"{starter} ")
            for starter in question_starters
        )
    
    def _process_question(self, user_input: str, language_info: Dict) -> Dict[str, Any]:
        """Process question using QA Agent"""
        results = {
            'session_id': self.session_id,
            'timestamp': datetime.datetime.now().isoformat(),
            'user_input': user_input,
            'language_detected': language_info,
            'type': 'question',
            'agent_results': {}
        }
        
        # Run QA directly so model loading does not race the background worker.
        qa_task = {'question': user_input, 'context': ''}
        qa_result = self.agents['qa_agent'].process_task(qa_task)
        self.shared_memory.set("QA Agent_result", qa_result)
        results['agent_results']['qa_agent'] = qa_result
        results['final_response'] = qa_result
        
        # Save to conversation history
        agent_response = qa_result.get('answer', '') if qa_result else 'No response'
        self.conversation_history.save_conversation(
            self.session_id,
            user_input,
            agent_response,
            'QA Agent',
            language_info.get('language_name', 'Unknown')
        )
        
        return results
    
    def _process_standard(self, user_input: str, language_info: Dict) -> Dict[str, Any]:
        """Process standard input through multi-agent system"""
        results = {
            'session_id': self.session_id,
            'timestamp': datetime.datetime.now().isoformat(),
            'user_input': user_input,
            'language_detected': language_info,
            'type': 'standard',
            'agent_results': {}
        }
        
        coordinator_task = {'id': 1, 'input': user_input}
        self.agents['coordinator'].assign_task(coordinator_task)
        
        import time
        time.sleep(0.5)
        
        analyzer_task = {'input': user_input}
        self.agents['analyzer'].assign_task(analyzer_task)
        
        researcher_task = {'topic': user_input[:50]}
        self.agents['researcher'].assign_task(researcher_task)
        
        time.sleep(1)
        
        validator_task = {'data': self.shared_memory.get_all()}
        self.agents['validator'].assign_task(validator_task)
        
        for agent_name in ['coordinator', 'analyzer', 'researcher', 'validator']:
            result = self.shared_memory.get(f"{self.agents[agent_name].name}_result")
            if result:
                results['agent_results'][agent_name] = result
        
        synthesizer_task = {'components': results['agent_results']}
        self.agents['synthesizer'].assign_task(synthesizer_task)
        
        time.sleep(0.5)
        
        synthesizer_result = self.shared_memory.get("Synthesizer_result")
        if synthesizer_result:
            results['final_response'] = synthesizer_result
        
        agent_response = json.dumps(results.get('final_response', {}), indent=2)
        self.conversation_history.save_conversation(
            self.session_id,
            user_input,
            agent_response,
            'MultiAgent',
            language_info.get('language_name', 'Unknown')
        )
        
        return results
    
    def display_results(self, results: Dict[str, Any]):
        """Display processing results"""
        print("\n" + "="*80)
        print("MULTI-AGENT SYSTEM RESULTS")
        print("="*80)
        
        print(f"\nSession ID: {results.get('session_id')}")
        print(f"Timestamp: {results.get('timestamp')}")
        print(f"Type: {results.get('type', 'standard').upper()}")
        print(f"Language: {results.get('language_detected', {}).get('language_name', 'Unknown')}")
        
        if results.get('type') == 'question':
            qa_result = results.get('agent_results', {}).get('qa_agent', {})
            if qa_result:
                print(f"\n--- Question ---")
                print(f"{results.get('user_input')}")
                print(f"\n--- Answer (GPT-2) ---")
                print(f"{qa_result.get('answer', 'N/A')}")
                print(f"\nModel: {qa_result.get('model', 'N/A')}")
        else:
            print("\n--- Agent Results ---")
            for agent_name, result in results.get('agent_results', {}).items():
                print(f"\n[{agent_name.upper()}]")
                print(json.dumps(result, indent=2)[:500])
            
            if 'final_response' in results:
                print("\n--- Final Response ---")
                final_response = results['final_response']
                print(f"Summary: {final_response.get('summary', 'N/A')}")
                print(f"Recommendations:")
                for rec in final_response.get('recommendations', []):
                    print(f"  - {rec}")
        
        print("\n" + "="*80)
    
    def get_session_history(self) -> List[Dict]:
        """Get current session history"""
        return self.conversation_history.get_session_history(self.session_id)
    
    def get_all_sessions(self) -> List[str]:
        """Get all session IDs"""
        return self.conversation_history.get_all_sessions()
    
    def view_history(self, session_id: str = None):
        """View conversation history"""
        if session_id is None:
            session_id = self.session_id
        
        history = self.conversation_history.get_session_history(session_id)
        
        print("\n" + "="*80)
        print(f"CONVERSATION HISTORY - Session: {session_id}")
        print("="*80)
        
        if not history:
            print("No conversations in this session.")
        else:
            for i, entry in enumerate(history, 1):
                print(f"\n[{i}] {entry['timestamp']}")
                print(f"Language: {entry['language']}")
                print(f"User: {entry['user_input']}")
                print(f"Agent ({entry['agent_role']}): {entry['agent_response'][:200]}...")
        
        print("\n" + "="*80)
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a specific session"""
        return self.conversation_history.delete_session(session_id)
    
    def clear_all_memory(self) -> bool:
        """Clear all conversations and shared memory"""
        try:
            self.shared_memory.clear()
            result = self.conversation_history.clear_all_conversations()
            self.agents['qa_agent'].clear_history()
            return result
        except Exception as e:
            print(f"Error clearing memory: {e}")
            return False
    
    def clear_session_memory(self, session_id: str = None) -> bool:
        """Clear memory for specific session"""
        if session_id is None:
            session_id = self.session_id
        return self.conversation_history.delete_session(session_id)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        return {
            'total_sessions': self.conversation_history.get_session_count(),
            'total_conversations': self.conversation_history.get_total_conversations(),
            'current_session_id': self.session_id,
            'current_session_history': len(self.get_session_history()),
            'shared_memory_size': len(self.shared_memory.get_all()),
            'qa_history_size': len(self.agents['qa_agent'].conversation_history)
        }


def interactive_menu(system: MultiAgentSystem):
    """Interactive menu for user"""
    while True:
        print("\n" + "="*80)
        print("MULTI-AGENT AI SYSTEM - INTERACTIVE MENU")
        print("="*80)
        print("1.  Text input / Ask question")
        print("2.  Speech input / Ask question")
        print("3.  View current session history")
        print("4.  View all sessions")
        print("5.  Load previous session")
        print("6.  Delete specific session")
        print("7.  View system memory")
        print("8.  Memory Statistics")
        print("9.  Clear all memory")
        print("10. Clear current session")
        print("11. Clear QA history")
        print("12. Exit")
        print("="*80)
        
        choice = input("Select option (1-12): ").strip()
        
        if choice == '1':
            text_input = input("\nEnter your text: ").strip()
            if text_input:
                results = system.process_input(user_input=text_input)
                system.display_results(results)
        
        elif choice == '2':
            if SPEECH_AVAILABLE:
                print("\nPrepare to speak...")
                results = system.process_input(use_speech=True)
                if 'error' not in results:
                    system.display_results(results)
                else:
                    print(f"Error: {results['error']}")
            else:
                print("Speech recognition not available. Install: pip install SpeechRecognition")
        
        elif choice == '3':
            system.view_history()
        
        elif choice == '4':
            sessions = system.get_all_sessions()
            print("\nAvailable Sessions:")
            if not sessions:
                print("No sessions found.")
            else:
                for i, session_id in enumerate(sessions, 1):
                    print(f"{i}. {session_id}")
        
        elif choice == '5':
            sessions = system.get_all_sessions()
            if sessions:
                print("\nAvailable Sessions:")
                for i, session_id in enumerate(sessions, 1):
                    print(f"{i}. {session_id}")
                try:
                    session_num = int(input("Select session number: ")) - 1
                    if 0 <= session_num < len(sessions):
                        system.view_history(sessions[session_num])
                except (ValueError, IndexError):
                    print("Invalid selection")
            else:
                print("No previous sessions found")
        
        elif choice == '6':
            sessions = system.get_all_sessions()
            if sessions:
                print("\nAvailable Sessions:")
                for i, session_id in enumerate(sessions, 1):
                    print(f"{i}. {session_id}")
                try:
                    session_num = int(input("Select session to delete: ")) - 1
                    if 0 <= session_num < len(sessions):
                        confirm = input(f"Delete session {sessions[session_num]}? (y/n): ").strip().lower()
                        if confirm == 'y':
                            if system.delete_session(sessions[session_num]):
                                print("✓ Session deleted")
                            else:
                                print("✗ Failed to delete session")
                except (ValueError, IndexError):
                    print("Invalid selection")
            else:
                print("No sessions to delete")
        
        elif choice == '7':
            memory = system.shared_memory.get_all()
            print("\nShared Memory Contents:")
            print(json.dumps(memory, indent=2, default=str))
        
        elif choice == '8':
            stats = system.get_memory_stats()
            print("\n" + "="*80)
            print("MEMORY STATISTICS")
            print("="*80)
            print(f"Total Sessions: {stats['total_sessions']}")
            print(f"Total Conversations: {stats['total_conversations']}")
            print(f"Current Session ID: {stats['current_session_id']}")
            print(f"Current Session History: {stats['current_session_history']} items")
            print(f"Shared Memory Size: {stats['shared_memory_size']} items")
            print(f"QA History Size: {stats['qa_history_size']} items")
            print("="*80)
        
        elif choice == '9':
            confirm = input("\nClear ALL memory and conversations? (yes/no): ").strip().lower()
            if confirm == 'yes':
                if system.clear_all_memory():
                    print("✓ All memory cleared")
                else:
                    print("✗ Failed to clear memory")
            else:
                print("Cancelled")
        
        elif choice == '10':
            confirm = input(f"\nClear current session ({system.session_id})? (yes/no): ").strip().lower()
            if confirm == 'yes':
                if system.clear_session_memory():
                    print("✓ Current session cleared")
                else:
                    print("✗ Failed to clear session")
            else:
                print("Cancelled")
        
        elif choice == '11':
            system.agents['qa_agent'].clear_history()
            print("✓ QA conversation history cleared")
        
        elif choice == '12':
            print("\nShutting down system...")
            system.stop_agents()
            break
        
        else:
            print("Invalid option. Please try again.")


def is_streamlit_runtime() -> bool:
    """Return True when this file is being executed by Streamlit."""
    if not STREAMLIT_AVAILABLE:
        return False

    try:
        streamlit_scriptrunner = importlib.import_module("streamlit.runtime.scriptrunner")
        get_script_run_ctx = streamlit_scriptrunner.get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


def get_streamlit_system() -> MultiAgentSystem:
    """Create or reuse the Streamlit session's multi-agent system."""
    if "multiagent_system" not in st.session_state:
        st.session_state.multiagent_system = MultiAgentSystem()
    return st.session_state.multiagent_system


def render_streamlit_result(result: Dict[str, Any]):
    """Render one agent response in Streamlit."""
    if "error" in result:
        st.error(result["error"])
        return

    language = result.get("language_detected", {}).get("language_name", "Unknown")
    st.caption(
        f"Session {result.get('session_id', 'N/A')} | "
        f"{result.get('type', 'standard').title()} | {language}"
    )

    if result.get("type") == "question":
        qa_result = result.get("agent_results", {}).get("qa_agent", {})
        st.markdown(qa_result.get("answer", "No answer returned."))
        st.caption(
            f"Model: {qa_result.get('model', 'N/A')} | "
            f"Confidence: {qa_result.get('confidence', 0):.2f}"
        )
        return

    final_response = result.get("final_response", {})
    if final_response:
        st.markdown(final_response.get("summary", "Processed by the multi-agent system."))
        recommendations = final_response.get("recommendations", [])
        if recommendations:
            st.markdown("**Recommendations**")
            for recommendation in recommendations:
                st.markdown(f"- {recommendation}")

    with st.expander("Agent details"):
        for agent_name, agent_result in result.get("agent_results", {}).items():
            st.markdown(f"**{agent_name.replace('_', ' ').title()}**")
            st.json(agent_result)


def render_streamlit_history(system: MultiAgentSystem, session_id: str):
    """Render persisted conversation history for a session."""
    history = system.conversation_history.get_session_history(session_id)
    if not history:
        st.info("No conversations in this session yet.")
        return

    for entry in history:
        with st.chat_message("user"):
            st.markdown(entry["user_input"])
            st.caption(f"{entry['timestamp']} | {entry['language']}")
        with st.chat_message("assistant"):
            st.markdown(entry["agent_response"])
            st.caption(entry["agent_role"])


def streamlit_app():
    """Streamlit UI for the multi-agent system."""
    if not STREAMLIT_AVAILABLE:
        print("Streamlit is not installed. Install it with: pip install streamlit")
        return

    st.set_page_config(
        page_title="Multi-Agent System",
        page_icon="MA",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .block-container { padding-block-start: 1.5rem; max-inline-size: 1180px; }
        [data-testid="stSidebar"] .stButton button { inline-size: 100%; }
        .stChatMessage { border-radius: 8px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    system = get_streamlit_system()

    with st.sidebar:
        st.title("Multi-Agent")
        st.caption(f"Current session: {system.session_id}")

        stats = system.get_memory_stats()
        col_a, col_b = st.columns(2)
        col_a.metric("Sessions", stats["total_sessions"])
        col_b.metric("Messages", stats["total_conversations"])
        st.metric("Shared memory", stats["shared_memory_size"])

        if st.button("New session", use_container_width=True):
            st.session_state.multiagent_system = MultiAgentSystem()
            st.session_state.last_result = None
            st.rerun()

        st.divider()

        sessions = system.get_all_sessions()
        selected_session = st.selectbox(
            "Session history",
            options=sessions or [system.session_id],
            index=0,
        )
        show_history = st.toggle("Show selected history", value=False)

        col_clear, col_delete = st.columns(2)
        if col_clear.button("Clear current"):
            if system.clear_session_memory():
                st.session_state.last_result = None
                st.toast("Current session cleared")
                st.rerun()

        if col_delete.button("Delete selected"):
            if system.delete_session(selected_session):
                st.toast("Session deleted")
                st.rerun()

        if st.button("Clear all memory", use_container_width=True):
            if system.clear_all_memory():
                st.session_state.last_result = None
                st.toast("All memory cleared")
                st.rerun()

        if st.button("Clear QA history", use_container_width=True):
            system.agents["qa_agent"].clear_history()
            st.toast("QA conversation history cleared")

        st.divider()
        st.caption(f"Speech: {'available' if SPEECH_AVAILABLE else 'not installed'}")
        st.caption(f"Browser voice input: {'available' if STREAMLIT_MIC_AVAILABLE else 'not installed'}")
        st.caption(f"Language detection: {'available' if LANGDETECT_AVAILABLE else 'not installed'}")
        st.caption(f"Local GPT-2: {'available' if LOCAL_AI_AVAILABLE else 'fallback only'}")

    st.title("Multi-Agent Workspace")

    if show_history:
        render_streamlit_history(system, selected_session)
    else:
        st.caption("Ask a question for the QA agent, or enter a task for the full multi-agent flow.")
        last_result = st.session_state.get("last_result")
        if last_result:
            with st.chat_message("user"):
                st.markdown(last_result.get("user_input", ""))
            with st.chat_message("assistant"):
                render_streamlit_result(last_result)
        else:
            st.info("Start by typing a question or task below.")

    st.divider()
    st.subheader("Voice Input")
    if STREAMLIT_MIC_AVAILABLE and speech_to_text:
        transcript = speech_to_text(
            start_prompt="Start recording",
            stop_prompt="Stop recording",
            just_once=False,
            use_container_width=True,
            language="en",
            key="voice_input",
        )

        if transcript:
            st.session_state.voice_transcript = transcript

        voice_text = st.text_area(
            "Transcript",
            value=st.session_state.get("voice_transcript", ""),
            placeholder="Record your voice, then review the transcript here.",
            height=90,
        )
        st.session_state.voice_transcript = voice_text

        col_send_voice, col_clear_voice = st.columns([1, 1])
        if col_send_voice.button("Send voice input", use_container_width=True, disabled=not voice_text.strip()):
            with st.spinner("Agents are working..."):
                st.session_state.last_result = system.process_input(user_input=voice_text.strip())
            st.session_state.voice_transcript = ""
            st.rerun()

        if col_clear_voice.button("Clear transcript", use_container_width=True):
            st.session_state.voice_transcript = ""
            st.rerun()
    else:
        st.warning("Browser voice input is not installed. Run: python -m pip install streamlit-mic-recorder")

    prompt = st.chat_input("Ask the agents...")
    if prompt:
        with st.spinner("Agents are working..."):
            st.session_state.last_result = system.process_input(user_input=prompt)
        st.rerun()


def main():
    """Main entry point"""
    print("\n" + "="*80)
    print("MULTI-AGENT AI SYSTEM WITH GPT-2")
    print("="*80)
    print("\nInitializing system...")
    
    # Check dependencies
    if not SPEECH_AVAILABLE:
        print(" Warning: Speech recognition not available")
        print(" Install with: pip install SpeechRecognition pydub")
    
    if not LANGDETECT_AVAILABLE:
        print(" Warning: Language detection not available")
        print(" Install with: pip install langdetect")
    
    if not LOCAL_AI_AVAILABLE:
        print(" Warning: GPT-2 dependencies not available")
        print(" Install with: pip install transformers torch")
    
    system = MultiAgentSystem()
    print("System initialized successfully")
    print(f" Session ID: {system.session_id}")
    print(f"GPT-2 AI: {'Available' if LOCAL_AI_AVAILABLE else 'Fallback only'}")
    
    try:
        interactive_menu(system)
    except KeyboardInterrupt:
        print("\n\nShutting down system...")
        system.stop_agents()
    except Exception as e:
        print(f"\nError: {e}")
        system.stop_agents()


if __name__ == "__main__":
    if is_streamlit_runtime():
        streamlit_app()
    else:
        main()
