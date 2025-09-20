#!/usr/bin/env python3
"""
Structured Conversation Analyzer V3
Two-level pipeline for deep analysis with "Life Balance Wheel"
"""

import os
import json
import zipfile
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
from tqdm.asyncio import tqdm_asyncio
import re
from collections import Counter

# OpenAI
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Visualization
import plotly.graph_objects as go
import plotly.express as px
import plotly.offline as pyo

load_dotenv()

class PersonalityTypeAnalyzer:
    """Analyzes communication patterns to determine 16 personalities types"""
    
    def __init__(self):
        # Keywords for analyzing MBTI dimensions
        self.extraversion_indicators = [
            'networking', 'meeting', 'collaboration', 'team', 'share', 'discuss', 'present'
        ]
        
        self.introversion_indicators = [
            'alone', 'think', 'analyze', 'study', 'individual', 'solo', 'personal'
        ]
        
        self.sensing_indicators = [
            'step by step', 'specific', 'practical', 'detailed', 'exact', 'instructions'
        ]
        
        self.intuition_indicators = [
            'possibilities', 'ideas', 'concept', 'theory', 'future', 'innovation', 'what if'
        ]
        
        self.thinking_indicators = [
            'logic', 'analysis', 'efficiency', 'objective', 'criteria', 'reason', 'pros and cons'
        ]
        
        self.feeling_indicators = [
            'people', 'feelings', 'values', 'harmony', 'motivation', 'relationships', 'ethical'
        ]
        
        self.judging_indicators = [
            'plan', 'schedule', 'deadline', 'organize', 'structure', 'order', 'finish'
        ]
        
        self.perceiving_indicators = [
            'flexible', 'adapt', 'options', 'open', 'spontaneous', 'explore', 'keep options'
        ]
        
        # Decision-making patterns
        self.fact_gathering_patterns = [
            'data', 'statistics', 'research', 'facts', 'information', 'sources', 'evidence'
        ]
        
        self.opportunity_seeking_patterns = [
            'alternatives', 'perspectives', 'trends', 'potential', 'opportunities'
        ]
        
        self.logical_analysis_patterns = [
            'analyze', 'compare', 'evaluate', 'weigh', 'rational', 'logical approach'
        ]
        
        self.human_factor_patterns = [
            'impact on people', 'team thinks', 'reaction', 'motivation', 'people feel'
        ]
        
        self.planning_patterns = [
            'action plan', 'phases', 'sequence', 'roadmap', 'strategy', 'systematic'
        ]
        
        self.improvisation_patterns = [
            'situational', 'adaptive', 'try', 'experiment', 'flexible approach', 'improvise'
        ]

    def analyze_communication_patterns(self, user_messages: list) -> dict:
        """Analyze communication patterns for MBTI"""
        if not user_messages:
            return self._get_default_mbti_scores()
        
        # Combine all user messages
        all_text = ' '.join(user_messages).lower()
        total_length = len(all_text)
        
        if total_length == 0:
            return self._get_default_mbti_scores()
        
        # Count indicators for each dimension
        e_score = self._count_indicators(all_text, self.extraversion_indicators)
        i_score = self._count_indicators(all_text, self.introversion_indicators)
        
        s_score = self._count_indicators(all_text, self.sensing_indicators)  
        n_score = self._count_indicators(all_text, self.intuition_indicators)
        
        t_score = self._count_indicators(all_text, self.thinking_indicators)
        f_score = self._count_indicators(all_text, self.feeling_indicators)
        
        j_score = self._count_indicators(all_text, self.judging_indicators)
        p_score = self._count_indicators(all_text, self.perceiving_indicators)
        
        # Additional metrics
        avg_message_length = sum(len(msg) for msg in user_messages) / len(user_messages)
        message_count = len(user_messages)
        
        # Normalization considering text length and additional factors
        return {
            'extraversion_score': self._normalize_personality_score(e_score, i_score, avg_message_length, 'extraversion'),
            'introversion_score': self._normalize_personality_score(i_score, e_score, avg_message_length, 'introversion'),
            'sensing_score': self._normalize_personality_score(s_score, n_score, message_count, 'sensing'),
            'intuition_score': self._normalize_personality_score(n_score, s_score, message_count, 'intuition'),
            'thinking_score': self._normalize_personality_score(t_score, f_score, total_length, 'thinking'),
            'feeling_score': self._normalize_personality_score(f_score, t_score, total_length, 'feeling'),
            'judging_score': self._normalize_personality_score(j_score, p_score, message_count, 'judging'),
            'perceiving_score': self._normalize_personality_score(p_score, j_score, message_count, 'perceiving'),
            'dominant_type': self._determine_dominant_type(e_score, i_score, s_score, n_score, t_score, f_score, j_score, p_score)
        }

    def analyze_decision_patterns(self, user_messages: list) -> dict:
        """Decision-making patterns analysis"""
        if not user_messages:
            return self._get_default_decision_scores()
        
        all_text = ' '.join(user_messages).lower()
        
        if not all_text.strip():
            return self._get_default_decision_scores()
        
        # Count decision-making patterns
        fact_gathering = self._count_indicators(all_text, self.fact_gathering_patterns)
        opportunity_seeking = self._count_indicators(all_text, self.opportunity_seeking_patterns)
        logical_analysis = self._count_indicators(all_text, self.logical_analysis_patterns)
        human_factor = self._count_indicators(all_text, self.human_factor_patterns)
        planning = self._count_indicators(all_text, self.planning_patterns)
        improvisation = self._count_indicators(all_text, self.improvisation_patterns)
        
        total_patterns = fact_gathering + opportunity_seeking + logical_analysis + human_factor + planning + improvisation
        
        if total_patterns == 0:
            return self._get_default_decision_scores()
        
        return {
            'fact_gathering_ratio': round((fact_gathering / total_patterns) * 10, 2),
            'opportunity_seeking_ratio': round((opportunity_seeking / total_patterns) * 10, 2),
            'logical_analysis_ratio': round((logical_analysis / total_patterns) * 10, 2),
            'human_factor_ratio': round((human_factor / total_patterns) * 10, 2),
            'planning_ratio': round((planning / total_patterns) * 10, 2),
            'improvisation_ratio': round((improvisation / total_patterns) * 10, 2),
            'decision_style': self._determine_decision_style(fact_gathering, opportunity_seeking, logical_analysis, human_factor, planning, improvisation)
        }

    def _count_indicators(self, text: str, indicators: list) -> int:
        """Count indicators in text"""
        count = 0
        for indicator in indicators:
            count += text.count(indicator.lower())
        return count

    def _normalize_personality_score(self, primary_score: int, opposite_score: int, context_metric: float, dimension: str) -> float:
        """Score normalization considering context"""
        if primary_score + opposite_score == 0:
            return 5.0  # Neutral score
        
        # Base score from 0 to 10
        base_score = (primary_score / (primary_score + opposite_score)) * 10
        
        # Adjustment based on contextual metrics
        if dimension == 'extraversion' and context_metric > 100:  # Long messages
            base_score = min(base_score + 1, 10)
        elif dimension == 'introversion' and context_metric < 50:  # Short messages
            base_score = min(base_score + 1, 10)
        elif dimension in ['sensing', 'judging'] and context_metric > 5:  # Many messages
            base_score = min(base_score + 0.5, 10)
        
        return round(base_score, 2)

    def _determine_dominant_type(self, e, i, s, n, t, f, j, p) -> str:
        """Determine dominant MBTI type"""
        type_code = ""
        type_code += "E" if e >= i else "I"
        type_code += "S" if s >= n else "N"  
        type_code += "T" if t >= f else "F"
        type_code += "J" if j >= p else "P"
        return type_code

    def _determine_decision_style(self, fact, opp, logic, human, plan, improv) -> str:
        """Determine decision-making style"""
        scores = {
            'Analytical': fact + logic,
            'Conceptual': opp + improv,
            'Directive': plan + logic,
            'Behavioral': human + plan
        }
        return max(scores.items(), key=lambda x: x[1])[0]

    def _get_default_mbti_scores(self) -> dict:
        """Default MBTI scores"""
        return {
            'extraversion_score': 5.0, 'introversion_score': 5.0,
            'sensing_score': 5.0, 'intuition_score': 5.0,
            'thinking_score': 5.0, 'feeling_score': 5.0,
            'judging_score': 5.0, 'perceiving_score': 5.0,
            'dominant_type': 'XXXX'
        }

    def _get_default_decision_scores(self) -> dict:
        """Default decision scores"""
        return {
            'fact_gathering_ratio': 2.5, 'opportunity_seeking_ratio': 2.5,
            'logical_analysis_ratio': 2.5, 'human_factor_ratio': 2.5,
            'planning_ratio': 2.5, 'improvisation_ratio': 2.5,
            'decision_style': 'Balanced'
        }

# --- Configuration ---
ANALYSIS_MODEL_L1 = "gpt-4.1-mini"
ANALYSIS_MODEL_L2 = "o3"
SAMPLE_SIZE = 10000                         # Full dataset - LLM topic analysis with caching
MAX_CONCURRENT_REQUESTS = 100          # Stable parallelization
RESULTS_FILE = "analysis_results_full.jsonl" # New file for full results
CLEAR_TOPIC_CACHE = False               # Full dataset now cached - future runs will be instant

class ConversationProcessor:
    """Level 1: Анализирует один диалог и возвращает structured JSON"""
    
    def __init__(self, client: AsyncOpenAI):
        self.client = client
        self.personality_analyzer = PersonalityTypeAnalyzer()
        self.life_balance_schema = {
            "career_professional": "Карьера и проф. развитие",
            "education_learning": "Образование и обучение",
            "health_fitness": "Здоровье и фитнес",
            "relationships_family": "Отношения и семья",
            "personal_finance": "Личные финансы",
            "hobbies_recreation": "Хобби и отдых",
            "spiritual_personal_growth": "Духовный и личностный рост",
            "social_community": "Общественная жизнь",
            "creativity_arts": "Творчество и искусство",
            "home_environment": "Дом и быт",
            "travel_adventure": "Путешествия и приключения",
            "emotional_wellbeing": "Эмоциональное состояние"
        }

    def _prepare_prompt(self, conversation_text: str) -> str:
        balance_categories = "\n".join([f"- {key} ({desc})" for key, desc in self.life_balance_schema.items()])
        
        return f"""
Проанализируй следующие сообщения пользователя в диалогах с ChatGPT. Верни ТОЛЬКО JSON объект.

СООБЩЕНИЯ ПОЛЬЗОВАТЕЛЯ:
---
{conversation_text[:4000]}
---

ЗАДАЧИ:
1.  **Content Analysis**: Определи тему, домен и сложность на основе запросов пользователя.
2.  **Communication Style**: Оцени стиль общения пользователя по шкале 0-10.
3.  **Life Balance Mapping**: Оцени, к каким сферам жизни относятся запросы пользователя.
4.  **Interest Detection**: Извлеки конкретные интересы, хобби, активности из запросов пользователя.
5.  **Behavioral Patterns**: Оцени тип сессии и намерения пользователя.

СФЕРЫ ЖИЗНИ:
{balance_categories}

JSON СХЕМА:
{{
  "content_analysis": {{
    "primary_topic": "string",
    "domain": "string (e.g., Technology, Health)",
    "complexity_level": "string (Beginner, Intermediate, Advanced, Expert)"
  }},
  "communication_style": {{
    "politeness_score": "float (0-10)",
    "technical_depth_score": "float (0-10)",
    "curiosity_score": "float (0-10)",
    "confidence_level": "float (0-10)",
    "formality_score": "float (0-10)"
  }},
  "life_balance_mapping": {{
    "career_professional": "float (0-10)", "education_learning": "float (0-10)",
    "health_fitness": "float (0-10)", "relationships_family": "float (0-10)",
    "personal_finance": "float (0-10)", "hobbies_recreation": "float (0-10)",
    "spiritual_personal_growth": "float (0-10)", "social_community": "float (0-10)",
    "creativity_arts": "float (0-10)", "home_environment": "float (0-10)",
    "travel_adventure": "float (0-10)", "emotional_wellbeing": "float (0-10)"
  }},
  "interests_detected": {{
    "sports": ["tennis", "football"] or [],
    "entertainment": ["tv_shows", "movies", "games"] or [],
    "technology": ["programming", "ai", "tools"] or [],
    "lifestyle": ["cooking", "fitness", "travel"] or [],
    "other": ["any_other_interests"] or []
  }},
  "session_analysis": {{
    "session_type": "string (quick_question, deep_exploration, problem_solving, learning, brainstorming)",
    "interaction_intensity": "float (0-10)",
    "goal_oriented": "boolean"
  }},
  "personality_patterns": {{
    "extraversion_score": "float (0-10)", "introversion_score": "float (0-10)",
    "sensing_score": "float (0-10)", "intuition_score": "float (0-10)",
    "thinking_score": "float (0-10)", "feeling_score": "float (0-10)",
    "judging_score": "float (0-10)", "perceiving_score": "float (0-10)",
    "dominant_type": "string (MBTI type like ENTJ, ISFP, etc.)"
  }},
  "decision_patterns": {{
    "fact_gathering_ratio": "float (0-10)", "opportunity_seeking_ratio": "float (0-10)",
    "logical_analysis_ratio": "float (0-10)", "human_factor_ratio": "float (0-10)",
    "planning_ratio": "float (0-10)", "improvisation_ratio": "float (0-10)",
    "decision_style": "string (Analytical, Conceptual, Directive, Behavioral)"
  }}
}}
"""

    def _truncate_long_message(self, message: str, max_length: int = 800) -> str:
        """Truncate long messages in start...end fashion"""
        if len(message) <= max_length:
            return message
        
        # Calculate how much to take from start and end
        start_length = max_length // 2 - 10  # Reserve space for "..."
        end_length = max_length - start_length - 10
        
        start_part = message[:start_length].strip()
        end_part = message[-end_length:].strip()
        
        return f"{start_part}...{end_part}"

    async def analyze(self, conversation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        user_messages = []
        mapping = conversation.get('mapping', {})
        
        for node in mapping.values():
            if (node and node.get('message') and 
                node['message'].get('author', {}).get('role') == 'user' and  # Only user messages
                node['message'].get('content', {}).get('content_type') == 'text'):
                parts = node['message']['content'].get('parts', [])
                if parts and parts[0].strip():
                    user_msg = parts[0].strip()
                    # Truncate if too long
                    truncated_msg = self._truncate_long_message(user_msg)
                    user_messages.append(f"USER: {truncated_msg}")
        
        if not user_messages:
            return None
        
        prompt = self._prepare_prompt("\n".join(user_messages))
        
        try:
            response = await self.client.chat.completions.create(
                model=ANALYSIS_MODEL_L1,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            analysis_data = json.loads(response.choices[0].message.content)
            analysis_data['conversation_id'] = conversation.get('id')
            
            # Extract clean user messages for personality analysis (without "USER:" prefix)
            clean_user_messages = []
            for msg in user_messages:
                if msg.startswith("USER: "):
                    clean_user_messages.append(msg[6:])  # Remove "USER: " prefix
                else:
                    clean_user_messages.append(msg)
            
            # Add personality analysis
            personality_patterns = self.personality_analyzer.analyze_communication_patterns(clean_user_messages)
            decision_patterns = self.personality_analyzer.analyze_decision_patterns(clean_user_messages)
            
            analysis_data['personality_patterns'] = personality_patterns
            analysis_data['decision_patterns'] = decision_patterns
            
            # Add temporal metadata for behavioral analysis
            create_time = datetime.fromtimestamp(conversation['create_time'])
            analysis_data['metadata'] = {
                'title': conversation.get('title', 'Без названия'),
                'date': create_time.isoformat(),
                'hour_of_day': create_time.hour,
                'day_of_week': create_time.weekday(),  # 0=Monday, 6=Sunday
                'week_number': create_time.isocalendar()[1],
                'month': create_time.month,
                'year': create_time.year,
                'model_used': conversation.get('default_model_slug', 'unknown'),
                'message_count': len(user_messages)
            }
            return analysis_data
        except Exception as e:
            print(f"Error analyzing conversation {conversation.get('id', 'unknown')}: {e}")
            return None

class PipelineManager:
    """Level 1: Управляет пайплайном анализа всех диалогов"""
    
    def __init__(self, zip_path: str):
        self.zip_path = zip_path
        # Оборачиваем клиент для трассировки
        self.client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.processor = ConversationProcessor(self.client)
    
    def _load_and_sample(self) -> List[Dict[str, Any]]:
        print(f"📁 Loading and sampling data (up to {SAMPLE_SIZE} conversations)...")
        with zipfile.ZipFile(self.zip_path, 'r') as zf:
            with zf.open('conversations.json') as f:
                all_convs = json.load(f)
        
        valid_convs = [c for c in all_convs if c.get('create_time')]
        valid_convs.sort(key=lambda x: x['create_time'], reverse=True)
        sample_convs = valid_convs[:SAMPLE_SIZE]
        
        print(f"✅ Selected {len(sample_convs)} conversations for analysis")
        return sample_convs

    async def run_analysis_pipeline(self):
        conversations = self._load_and_sample()
        print(f"🚀 Starting analysis of {len(conversations)} conversations using {ANALYSIS_MODEL_L1}...")
        
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

        async def process_with_semaphore(conv):
            async with semaphore:
                return await self.processor.analyze(conv)

        tasks = [process_with_semaphore(conv) for conv in conversations]
        results = []
        for f in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
            result = await f
            if result:
                results.append(result)

        with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"\n✅ Analysis completed. Results saved to {RESULTS_FILE}")
        print(f"   Successfully processed: {len(results)}/{len(conversations)} conversations")

class AnalyticsEngine:
    """Level 2: Агрегирует данные и выполняет запросы"""
    
    def __init__(self, results_file: str):
        self.results_file = results_file
        self.df = self._load_data()
        self.life_balance_keys = list(ConversationProcessor(None).life_balance_schema.keys())

    def _load_data(self) -> pd.DataFrame:
        print(f"📊 Loading and aggregating data from {self.results_file}...")
        records = []
        
        # Check if file exists and has content
        if not os.path.exists(self.results_file):
            raise FileNotFoundError(f"Файл результатов {self.results_file} не найден")
        
        with open(self.results_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                try:
                    data = json.loads(line)
                    flat_data = {
                        'id': data['conversation_id'],
                        'date': pd.to_datetime(data['metadata']['date']),
                        'title': data['metadata']['title'],
                        'hour_of_day': data['metadata']['hour_of_day'],
                        'day_of_week': data['metadata']['day_of_week'],
                        'week_number': data['metadata']['week_number'],
                        'month': data['metadata']['month'],
                        'year': data['metadata']['year'],
                        'message_count': data['metadata']['message_count'],
                        'topic': data['content_analysis']['primary_topic'],
                        'domain': data['content_analysis']['domain'],
                        'complexity': data['content_analysis']['complexity_level'],
                        'politeness': float(data['communication_style']['politeness_score']),
                        'technical_depth': float(data['communication_style']['technical_depth_score']),
                        'curiosity': float(data['communication_style']['curiosity_score']),
                        'confidence': float(data['communication_style'].get('confidence_level', 5.0)),
                        'formality': float(data['communication_style'].get('formality_score', 5.0)),
                        'session_type': data.get('session_analysis', {}).get('session_type', 'unknown'),
                        'interaction_intensity': float(data.get('session_analysis', {}).get('interaction_intensity', 5.0)),
                        'goal_oriented': data.get('session_analysis', {}).get('goal_oriented', False),
                    }
                    
                    # Add personality patterns
                    personality = data.get('personality_patterns', {})
                    flat_data.update({
                        'extraversion_score': float(personality.get('extraversion_score', 5.0)),
                        'introversion_score': float(personality.get('introversion_score', 5.0)),
                        'sensing_score': float(personality.get('sensing_score', 5.0)),
                        'intuition_score': float(personality.get('intuition_score', 5.0)),
                        'thinking_score': float(personality.get('thinking_score', 5.0)),
                        'feeling_score': float(personality.get('feeling_score', 5.0)),
                        'judging_score': float(personality.get('judging_score', 5.0)),
                        'perceiving_score': float(personality.get('perceiving_score', 5.0)),
                        'dominant_type': personality.get('dominant_type', 'XXXX')
                    })
                    
                    # Add decision patterns
                    decisions = data.get('decision_patterns', {})
                    flat_data.update({
                        'fact_gathering_ratio': float(decisions.get('fact_gathering_ratio', 2.5)),
                        'opportunity_seeking_ratio': float(decisions.get('opportunity_seeking_ratio', 2.5)),
                        'logical_analysis_ratio': float(decisions.get('logical_analysis_ratio', 2.5)),
                        'human_factor_ratio': float(decisions.get('human_factor_ratio', 2.5)),
                        'planning_ratio': float(decisions.get('planning_ratio', 2.5)),
                        'improvisation_ratio': float(decisions.get('improvisation_ratio', 2.5)),
                        'decision_style': decisions.get('decision_style', 'Balanced')
                    })
                    
                    # Add life balance mapping
                    for key, value in data['life_balance_mapping'].items():
                        flat_data[key] = float(value)
                    
                    # Add interests detection (flatten lists into strings for easier analysis)
                    interests = data.get('interests_detected', {})
                    for category, items in interests.items():
                        flat_data[f'interests_{category}'] = ', '.join(items) if items else ''
                    records.append(flat_data)
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    print(f"Skipping invalid line: {e}")
                    continue
        
        if not records:
            raise ValueError(f"Файл {self.results_file} не содержит валидных данных. Возможно, анализ не был выполнен успешно.")
        
        df = pd.DataFrame(records)
        df.set_index('date', inplace=True)
        print(f"✅ Data aggregated. {len(df)} valid records.")
        return df

    def get_life_balance_wheel(self):
        balance_scores = self.df[self.life_balance_keys].mean()
        return balance_scores

    def get_balance_evolution(self, freq: str = 'ME'):
        evolution = self.df[self.life_balance_keys].resample(freq).mean().dropna()
        return evolution
    
    def get_neglected_areas(self, threshold: float = 2.0):
        balance_scores = self.get_life_balance_wheel()
        neglected = balance_scores[balance_scores < threshold]
        return neglected.sort_values()

    def get_personality_type_distribution(self):
        """Получить распределение MBTI типов"""
        type_counts = self.df['dominant_type'].value_counts()
        return type_counts
    
    def get_mbti_dimensions_averages(self):
        """Получить средние скоры по измерениям MBTI"""
        mbti_cols = ['extraversion_score', 'introversion_score', 'sensing_score', 'intuition_score',
                     'thinking_score', 'feeling_score', 'judging_score', 'perceiving_score']
        return self.df[mbti_cols].mean()
    
    def get_decision_style_distribution(self):
        """Получить распределение стилей принятия решений"""
        style_counts = self.df['decision_style'].value_counts()
        return style_counts
    
    def get_decision_patterns_averages(self):
        """Получить средние значения паттернов принятия решений"""
        decision_cols = ['fact_gathering_ratio', 'opportunity_seeking_ratio', 'logical_analysis_ratio',
                        'human_factor_ratio', 'planning_ratio', 'improvisation_ratio']
        return self.df[decision_cols].mean()
    
    def get_personality_evolution(self, freq='ME'):
        """Отследить эволюцию personality паттернов во времени"""
        mbti_cols = ['extraversion_score', 'introversion_score', 'sensing_score', 'intuition_score',
                     'thinking_score', 'feeling_score', 'judging_score', 'perceiving_score']
        personality_evolution = self.df[mbti_cols].resample(freq).mean().dropna()
        return personality_evolution
    
    def get_decision_evolution(self, freq='ME'):
        """Отследить эволюцию паттернов принятия решений во времени"""
        decision_cols = ['fact_gathering_ratio', 'opportunity_seeking_ratio', 'logical_analysis_ratio',
                        'human_factor_ratio', 'planning_ratio', 'improvisation_ratio']
        decision_evolution = self.df[decision_cols].resample(freq).mean().dropna()
        return decision_evolution
    
    def get_personality_correlations(self):
        """Получить корреляции между personality чертами и другими метриками"""
        personality_cols = ['extraversion_score', 'sensing_score', 'thinking_score', 'judging_score']
        other_cols = ['politeness', 'technical_depth', 'curiosity', 'confidence', 'formality']
        
        correlations = {}
        for p_col in personality_cols:
            if p_col in self.df.columns:
                correlations[p_col] = {}
                for o_col in other_cols:
                    if o_col in self.df.columns:
                        corr = self.df[p_col].corr(self.df[o_col])
                        correlations[p_col][o_col] = round(corr, 3) if not pd.isna(corr) else 0
        
        return correlations

class TemporalAnalyzer:
    """Analyzes behavioral patterns and usage trends over time"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
    
    def get_daily_usage_pattern(self):
        """Returns average usage by hour of day"""
        return self.df.groupby('hour_of_day').size()
    
    def get_weekly_usage_pattern(self):
        """Returns average usage by day of week"""
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        weekly = self.df.groupby('day_of_week').size()
        weekly.index = [day_names[i] for i in weekly.index]
        return weekly
    
    def get_communication_style_evolution(self, freq='ME'):
        """Track how communication style changed over time"""
        style_cols = ['politeness', 'technical_depth', 'curiosity', 'confidence', 'formality']
        return self.df[style_cols].resample(freq).mean().dropna()
    
    def get_session_type_trends(self, freq='ME'):
        """Track session type preferences over time"""
        session_trends = self.df.groupby([pd.Grouper(freq=freq), 'session_type']).size().unstack(fill_value=0)
        return session_trends
    
    def get_complexity_evolution(self, freq='ME'):
        """Track how conversation complexity evolved"""
        complexity_mapping = {'Beginner': 1, 'Intermediate': 2, 'Advanced': 3, 'Expert': 4}
        self.df['complexity_numeric'] = self.df['complexity'].map(complexity_mapping)
        return self.df['complexity_numeric'].resample(freq).mean().dropna()

class LLMTopicDiscovery:
    """LLM-Powered Topic Discovery Pipeline - Phase 1: Topic Extraction"""
    
    def __init__(self, client: AsyncOpenAI):
        self.client = client
        self.discovered_topics = {}  # Will store all discovered topics
        self.topic_taxonomy = {}     # Will store hierarchical organization
        
    async def extract_topics_from_conversation(self, title: str, user_messages: str) -> List[str]:
        """Phase 1: Extract topics from a single conversation using LLM"""
        
        prompt = f"""
Analyze this conversation and extract 3-5 specific topics, interests, or activities that the user is discussing or asking about.

CONVERSATION TITLE: {title}
USER MESSAGES: {user_messages[:1000]}

Return ONLY a JSON list of specific topics. Be precise and specific. Examples:
- Instead of "sports" say "tennis" or "football" 
- Instead of "technology" say "Python programming" or "iPhone photography"
- Instead of "entertainment" say "Netflix series" or "indie music"

Focus on the user's actual interests, hobbies, activities, or subjects they care about.

JSON format: ["topic1", "topic2", "topic3"]
"""
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            topics = result.get('topics', []) if isinstance(result, dict) else result
            return [topic.strip().lower() for topic in topics if topic.strip()]
            
        except Exception as e:
            print(f"Topic extraction error: {e}")
            return []
    
    async def consolidate_topics(self, all_topics: List[str]) -> Dict[str, Any]:
        """Phase 2: Consolidate and organize topics using LLM"""
        
        # Count topic frequency
        topic_counts = {}
        for topic in all_topics:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        # Get unique topics sorted by frequency
        unique_topics = sorted(topic_counts.keys(), key=lambda x: topic_counts[x], reverse=True)
        
        if len(unique_topics) < 3:
            return {"categories": {}, "topic_mapping": topic_counts}
        
        # Take top topics for consolidation (limit to avoid token limits)
        top_topics = unique_topics[:50]
        
        prompt = f"""
Analyze these topics extracted from user conversations and organize them intelligently:

TOPICS: {json.dumps(top_topics)}

Your tasks:
1. Group similar/related topics together
2. Create logical category names for each group  
3. Identify which topics are essentially the same (synonyms/variations)

IMPORTANT: Use only ASCII characters in your response. Avoid any unicode, emoji, or special characters.

Return JSON with this structure:
{{
  "categories": {{
    "category_name": ["topic1", "topic2", ...],
    "another_category": ["topic3", "topic4", ...]
  }},
  "synonyms": {{
    "canonical_topic": ["variation1", "variation2", ...],
    "another_canonical": ["var1", "var2", ...]
  }}
}}

Examples:
- Group "tennis", "tennis training", "tennis matches" -> "tennis"
- "python", "python programming", "coding in python" -> synonyms of "python programming"
"""
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            # Clean the response content to handle invalid unicode escapes
            content = response.choices[0].message.content
            
            # Aggressive JSON cleaning to handle unicode issues
            import re
            
            # Strategy: Clean content step by step
            try:
                # First try normal parsing
                result = json.loads(content)
            except json.JSONDecodeError as first_error:
                try:
                    # Remove all problematic unicode sequences
                    cleaned = content
                    # Remove incomplete unicode escapes
                    cleaned = re.sub(r'\\u[0-9a-fA-F]{0,3}(?![0-9a-fA-F])', '', cleaned)
                    # Remove malformed unicode 
                    cleaned = re.sub(r'\\u[^"0-9a-fA-F][^"]*', '', cleaned)
                    # Remove any remaining backslash-u followed by non-hex
                    cleaned = re.sub(r'\\u[^0-9a-fA-F][^"]*', '', cleaned)
                    result = json.loads(cleaned)
                except json.JSONDecodeError as second_error:
                    try:
                        # More aggressive: remove all backslash-u sequences
                        cleaned = re.sub(r'\\u[^"]*', '', content)
                        result = json.loads(cleaned)
                    except json.JSONDecodeError as third_error:
                        print(f"JSON parsing failed after multiple attempts: {first_error}")
                        # Return minimal valid structure
                        result = {"categories": {}, "synonyms": {}}
            
            # Build canonical topic mapping
            canonical_mapping = {}
            
            # Handle synonyms
            if "synonyms" in result:
                for canonical, variations in result["synonyms"].items():
                    canonical_mapping[canonical.lower()] = canonical.lower()
                    for var in variations:
                        canonical_mapping[var.lower()] = canonical.lower()
            
            # Handle categories
            organized_topics = {
                "categories": result.get("categories", {}),
                "canonical_mapping": canonical_mapping,
                "topic_counts": topic_counts
            }
            
            return organized_topics
            
        except Exception as e:
            print(f"Topic consolidation error: {e}")
            return {"categories": {}, "canonical_mapping": {}, "topic_counts": topic_counts}

class PersonalEvolutionAnalyzer:
    """Deep analysis of personal evolution: then vs now transformation"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.early_period = None
        self.recent_period = None
        self.evolution_insights = {}
        self._segment_temporal_periods()
    
    def _segment_temporal_periods(self):
        """Split conversations into early vs recent periods"""
        # Sort by date
        sorted_df = self.df.sort_index()
        
        # Define periods: first 6 months vs last 6 months
        total_span = sorted_df.index.max() - sorted_df.index.min()
        six_months = pd.Timedelta(days=180)
        
        # Early period: first 6 months
        early_cutoff = sorted_df.index.min() + six_months
        self.early_period = sorted_df[sorted_df.index <= early_cutoff]
        
        # Recent period: last 6 months  
        recent_cutoff = sorted_df.index.max() - six_months
        self.recent_period = sorted_df[sorted_df.index >= recent_cutoff]
        
        print(f"📊 Evolution Analysis Periods:")
        print(f"  Early: {self.early_period.index.min().strftime('%Y-%m-%d')} to {early_cutoff.strftime('%Y-%m-%d')} ({len(self.early_period)} conversations)")
        print(f"  Recent: {recent_cutoff.strftime('%Y-%m-%d')} to {self.recent_period.index.max().strftime('%Y-%m-%d')} ({len(self.recent_period)} conversations)")
    
    def calculate_sophistication_metrics(self) -> Dict[str, Any]:
        """Calculate communication and thinking sophistication metrics"""
        
        def analyze_period(period_df: pd.DataFrame, period_name: str) -> Dict[str, float]:
            if len(period_df) == 0:
                return {}
                
            metrics = {}
            
            # 1. Communication Sophistication
            titles = period_df['title'].fillna('').astype(str)
            
            # Average question length
            metrics['avg_question_length'] = titles.str.len().mean()
            
            # Vocabulary sophistication (unique words per conversation)
            metrics['vocabulary_richness'] = titles.apply(lambda x: len(set(x.lower().split()))).mean()
            
            # Question type distribution
            how_questions = titles.str.contains('how', case=False, na=False).sum()
            why_questions = titles.str.contains('why', case=False, na=False).sum()
            what_questions = titles.str.contains('what', case=False, na=False).sum()
            
            total_questions = len(titles)
            metrics['how_question_ratio'] = how_questions / max(total_questions, 1)
            metrics['why_question_ratio'] = why_questions / max(total_questions, 1)
            metrics['what_question_ratio'] = what_questions / max(total_questions, 1)
            
            # Confidence markers
            uncertain_markers = ['how do i', 'how to', 'i don\'t know', 'help me', 'i need']
            confident_markers = ['what do you think', 'my approach', 'i believe', 'i think', 'my hypothesis']
            
            uncertain_count = sum(titles.str.contains(marker, case=False, na=False).sum() for marker in uncertain_markers)
            confident_count = sum(titles.str.contains(marker, case=False, na=False).sum() for marker in confident_markers)
            
            metrics['uncertainty_ratio'] = uncertain_count / max(total_questions, 1)
            metrics['confidence_ratio'] = confident_count / max(total_questions, 1)
            
            # 2. Problem Complexity
            complex_words = ['system', 'strategy', 'framework', 'optimization', 'architecture', 'methodology', 'paradigm']
            simple_words = ['fix', 'bug', 'error', 'problem', 'issue']
            
            complex_count = sum(titles.str.contains(word, case=False, na=False).sum() for word in complex_words)
            simple_count = sum(titles.str.contains(word, case=False, na=False).sum() for word in simple_words)
            
            metrics['complexity_ratio'] = complex_count / max(total_questions, 1)
            metrics['simplicity_ratio'] = simple_count / max(total_questions, 1)
            
            # 3. Scope and Planning
            immediate_words = ['now', 'today', 'quick', 'fast', 'urgent']
            strategic_words = ['future', 'long-term', 'plan', 'strategy', 'roadmap', 'vision']
            
            immediate_count = sum(titles.str.contains(word, case=False, na=False).sum() for word in immediate_words)
            strategic_count = sum(titles.str.contains(word, case=False, na=False).sum() for word in strategic_words)
            
            metrics['immediate_focus_ratio'] = immediate_count / max(total_questions, 1)
            metrics['strategic_focus_ratio'] = strategic_count / max(total_questions, 1)
            
            # 4. Communication Style Metrics (from existing data)
            style_cols = ['politeness', 'formality', 'confidence', 'curiosity', 'technical_depth']
            for col in style_cols:
                if col in period_df.columns:
                    metrics[f'avg_{col}'] = period_df[col].mean()
            
            # 5. Life Balance Focus
            balance_cols = [col for col in period_df.columns if col.startswith('balance_')]
            if balance_cols:
                metrics['life_balance_diversity'] = period_df[balance_cols].gt(0).sum(axis=1).mean()
                metrics['avg_life_balance'] = period_df[balance_cols].mean().mean()
            
            return metrics
        
        early_metrics = analyze_period(self.early_period, "Early")
        recent_metrics = analyze_period(self.recent_period, "Recent")
        
        return {
            "early": early_metrics,
            "recent": recent_metrics,
            "evolution": self._calculate_evolution_scores(early_metrics, recent_metrics)
        }
    
    def _calculate_evolution_scores(self, early: Dict[str, float], recent: Dict[str, float]) -> Dict[str, float]:
        """Calculate evolution scores showing direction and magnitude of change"""
        evolution = {}
        
        for key in early.keys():
            if key in recent and early[key] > 0:
                # Calculate percentage change
                change = ((recent[key] - early[key]) / early[key]) * 100
                evolution[f"{key}_change"] = change
                evolution[f"{key}_direction"] = "increase" if change > 0 else "decrease"
                evolution[f"{key}_magnitude"] = abs(change)
        
        return evolution
    
    async def generate_evolution_insights(self, sophistication_metrics: Dict[str, Any], client: AsyncOpenAI) -> Dict[str, Any]:
        """Generate deep insights about personal transformation using LLM"""
        
        # Prepare data for LLM analysis
        early_sample = self.early_period['title'].dropna().sample(min(20, len(self.early_period))).tolist()
        recent_sample = self.recent_period['title'].dropna().sample(min(20, len(self.recent_period))).tolist()
        
        prompt = f"""
Analyze this person's intellectual and personal evolution over 2 years of ChatGPT conversations.

EARLY PERIOD QUESTIONS (first 6 months):
{json.dumps(early_sample, indent=2)}

RECENT PERIOD QUESTIONS (last 6 months):
{json.dumps(recent_sample, indent=2)}

QUANTITATIVE CHANGES:
{json.dumps(sophistication_metrics, indent=2)}

Provide deep insights about their transformation in JSON format:
{{
  "cognitive_evolution": {{
    "thinking_complexity": "How has their thinking become more sophisticated?",
    "problem_solving_approach": "How has their approach to problems evolved?",
    "learning_style": "How has their learning and question-asking style changed?"
  }},
  "personality_development": {{
    "confidence_growth": "How has their confidence and self-assurance evolved?",
    "intellectual_curiosity": "How has their curiosity and exploration changed?",
    "communication_style": "How has their communication become more mature?"
  }},
  "life_focus_evolution": {{
    "priority_shifts": "What priority changes are evident?",
    "scope_expansion": "How has their scope of concern expanded?",
    "time_horizon": "How has their planning and thinking timeframe evolved?"
  }},
  "growth_trajectory": {{
    "biggest_changes": "What are the 3 most significant transformations?",
    "growth_areas": "Where do they show the most development?",
    "future_direction": "What does their trajectory suggest about continued growth?"
  }},
  "psychological_profile": {{
    "then_vs_now": "Key psychological differences between early and recent periods",
    "maturation_signs": "Evidence of intellectual and emotional maturation",
    "emerging_strengths": "New strengths and capabilities that have emerged"
  }}
}}

Be specific, insightful, and focus on meaningful patterns of personal development.
"""
        
        try:
            # Use the lighter model for this analysis
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            insights = json.loads(content)
            return insights
            
        except Exception as e:
            print(f"Evolution insights generation error: {e}")
            return {
                "cognitive_evolution": {"error": "Analysis failed"},
                "personality_development": {"error": "Analysis failed"},
                "life_focus_evolution": {"error": "Analysis failed"},
                "growth_trajectory": {"error": "Analysis failed"},
                "psychological_profile": {"error": "Analysis failed"}
            }


class InterestTrendAnalyzer:
    """Analyzes interest evolution and trending topics using LLM discovery"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.interest_columns = [col for col in df.columns if col.startswith('interests_')]
        self.topic_discovery = None  # Will be set when needed
    
    def extract_all_interests(self):
        """Extract and count all unique interests"""
        all_interests = {}
        for col in self.interest_columns:
            category = col.replace('interests_', '')
            for interest_list in self.df[col].dropna():
                if interest_list.strip():
                    interests = [i.strip() for i in interest_list.split(',') if i.strip()]
                    for interest in interests:
                        if interest not in all_interests:
                            all_interests[interest] = {'category': category, 'count': 0, 'conversations': []}
                        all_interests[interest]['count'] += 1
        return all_interests
    
    def get_interest_trends(self, top_n=10, freq='ME'):
        """Track top interests over time"""
        all_interests = self.extract_all_interests()
        top_interests = sorted(all_interests.keys(), key=lambda x: all_interests[x]['count'], reverse=True)[:top_n]
        
        trends = {}
        for interest in top_interests:
            # Count conversations containing this interest by time period
            interest_mask = pd.Series(False, index=self.df.index)
            for col in self.interest_columns:
                interest_mask |= self.df[col].str.contains(interest, case=False, na=False)
            trends[interest] = interest_mask.resample(freq).sum()
        
        return pd.DataFrame(trends).fillna(0)
    
    def get_seasonal_interests(self):
        """Identify seasonal patterns in interests"""
        seasonal_data = {}
        for col in self.interest_columns:
            category = col.replace('interests_', '')
            # Count non-empty interest entries by month
            monthly_counts = self.df[self.df[col] != ''].groupby('month').size()
            seasonal_data[category] = monthly_counts
        return pd.DataFrame(seasonal_data).fillna(0)
    
    def get_emerging_interests(self, recent_months=6):
        """Identify interests that became popular recently"""
        cutoff_date = self.df.index.max() - pd.DateOffset(months=recent_months)
        recent_df = self.df[self.df.index >= cutoff_date]
        older_df = self.df[self.df.index < cutoff_date]
        
        recent_interests = InterestTrendAnalyzer(recent_df).extract_all_interests()
        older_interests = InterestTrendAnalyzer(older_df).extract_all_interests()
        
        emerging = {}
        for interest, data in recent_interests.items():
            old_count = older_interests.get(interest, {}).get('count', 0)
            if data['count'] > old_count * 2:  # At least 2x increase
                emerging[interest] = {
                    'recent_count': data['count'],
                    'old_count': old_count,
                    'growth_factor': data['count'] / max(old_count, 1)
                }
        
        return emerging
    
    async def discover_topics_from_conversations(self, client: AsyncOpenAI) -> Dict[str, Any]:
        """Phase 1 & 2: Discover and organize topics using LLM"""
        
        print(f"🔍 Starting topic discovery for {len(self.df)} total conversations")
        
        # Check for cached Phase 1 results
        cache_file = "topics_phase1_cache.json"
        topics_cache_file = "topics_extracted_cache.json"
        
        # Clear cache if requested
        if CLEAR_TOPIC_CACHE:
            for file in [cache_file, topics_cache_file]:
                if os.path.exists(file):
                    os.remove(file)
                    print(f"🗑️ Cleared cache file: {file}")
        
        # Always initialize topic_discovery for Phase 2
        topic_discovery = LLMTopicDiscovery(client)
        all_topics = []
        conversation_topics = {}
        
        if os.path.exists(cache_file) and os.path.exists(topics_cache_file) and not CLEAR_TOPIC_CACHE:
            print("📂 Found cached Phase 1 results, loading...")
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_topics = json.load(f)
                with open(topics_cache_file, 'r', encoding='utf-8') as f:
                    all_topics = json.load(f)
                    
                # Convert string keys back to match DataFrame index format
                conversation_topics = {}
                for str_key, topics in cached_topics.items():
                    # Try to find matching index in DataFrame
                    for idx in self.df.index:
                        if str(idx) == str_key:
                            conversation_topics[idx] = topics
                            break
                
                print(f"✅ Loaded {len(all_topics)} cached topics from {len(conversation_topics)} conversations")
                print(f"📊 Cache covers {len(conversation_topics)} conversations vs {len(self.df)} total available")
            except Exception as e:
                print(f"⚠️ Cache loading failed: {e}, running fresh analysis...")
                all_topics = []
                conversation_topics = {}
        else:
            print("🔍 Phase 1: Extracting topics from conversations using LLM...")
        
        # Set up sample data
        sample_size = len(self.df)  # Use all processed conversations
        sample_df = self.df
        
        # Only run extraction if we don't have cached data
        if not all_topics:
            # Extract topics from each conversation
            semaphore = asyncio.Semaphore(20)  # Increased concurrent requests for larger datasets
            
            async def extract_for_conversation(row):
                async with semaphore:
                    title = row['title']
                    # Reconstruct user messages from the conversation
                    user_msg_context = f"Title: {title}"
                    topics = await topic_discovery.extract_topics_from_conversation(title, user_msg_context)
                    return topics, row.name
            
            tasks = [extract_for_conversation(row) for _, row in sample_df.iterrows()]
            
            for i, task in enumerate(asyncio.as_completed(tasks)):
                topics, conv_id = await task
                all_topics.extend(topics)
                conversation_topics[conv_id] = topics
                
                if (i + 1) % 20 == 0:
                    print(f"  Processed {i + 1}/{len(tasks)} conversations...")
            
            print(f"✅ Extracted {len(all_topics)} total topics from {len(tasks)} conversations")
            
            # Save Phase 1 results to cache
            try:
                # Convert keys to strings for JSON serialization
                conversation_topics_serializable = {str(k): v for k, v in conversation_topics.items()}
                
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(conversation_topics_serializable, f, ensure_ascii=False, indent=2)
                with open(topics_cache_file, 'w', encoding='utf-8') as f:
                    json.dump(all_topics, f, ensure_ascii=False, indent=2)
                print("💾 Phase 1 results cached for future runs")
            except Exception as e:
                print(f"⚠️ Failed to cache results: {e}")
        
        if not all_topics:
            return {"organized_topics": {}, "conversation_topics": {}}
        
        print("🔄 Phase 2: Consolidating and organizing topics...")
        print(f"📊 Processing {len(all_topics)} total topic mentions from {len(conversation_topics)} conversations")
        
        # Consolidate topics
        organized_topics = await topic_discovery.consolidate_topics(all_topics)
        
        categories = organized_topics.get('categories', {})
        topic_counts = organized_topics.get('topic_counts', {})
        
        print(f"✅ Organized into {len(categories)} categories")
        print(f"📈 Topic frequency stats:")
        print(f"  - Unique topics: {len(topic_counts)}")
        print(f"  - Total mentions: {sum(topic_counts.values())}")
        if topic_counts:
            top_3 = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"  - Top topics: {[f'{topic}({count})' for topic, count in top_3]}")
        
        return {
            "organized_topics": organized_topics,
            "conversation_topics": conversation_topics
        }
    
    def get_sports_topics(self, discovered_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and analyze sports-related topics from discovered data"""
        organized_topics = discovered_data.get("organized_topics", {})
        categories = organized_topics.get("categories", {})
        topic_counts = organized_topics.get("topic_counts", {})
        
        # Find sports-related categories and topics
        sports_keywords = ['sport', 'game', 'play', 'team', 'match', 'tournament', 'league', 
                          'football', 'tennis', 'basketball', 'soccer', 'baseball', 'hockey',
                          'golf', 'swimming', 'running', 'fitness', 'workout', 'exercise',
                          'athlete', 'competition', 'championship', 'olympic']
        
        sports_topics = {}
        sports_categories = []
        
        # Check categories for sports content
        for category, topics in categories.items():
            if any(keyword in category.lower() for keyword in sports_keywords):
                sports_categories.append(category)
                for topic in topics:
                    if topic in topic_counts:
                        sports_topics[topic] = topic_counts[topic]
        
        # Also check individual topics for sports keywords
        for topic, count in topic_counts.items():
            if any(keyword in topic.lower() for keyword in sports_keywords):
                if topic not in sports_topics:
                    sports_topics[topic] = count
        
        # Get top sports topics
        top_sports = sorted(sports_topics.items(), key=lambda x: x[1], reverse=True)[:15]
        
        return {
            "sports_topics": dict(top_sports),
            "sports_categories": sports_categories,
            "total_sports_mentions": sum(sports_topics.values())
        }
    
    def get_language_trends(self) -> Dict[str, Any]:
        """Detect main language trends in conversations over time"""
        
        def detect_language_simple(text: str) -> str:
            """Simple language detection based on character patterns"""
            if not text:
                return "unknown"
            
            # Count different character types
            cyrillic_chars = len(re.findall(r'[а-яё]', text.lower()))
            latin_chars = len(re.findall(r'[a-z]', text.lower()))
            total_chars = cyrillic_chars + latin_chars
            
            if total_chars == 0:
                return "unknown"
            
            cyrillic_ratio = cyrillic_chars / total_chars
            
            if cyrillic_ratio > 0.3:  # More than 30% cyrillic
                return "russian"
            elif latin_chars > cyrillic_chars:
                return "english"
            else:
                return "mixed"
        
        # Apply language detection to titles
        languages = []
        dates = []
        
        for idx, row in self.df.iterrows():
            title = row.get('title', '')
            lang = detect_language_simple(title)
            languages.append(lang)
            dates.append(idx)
        
        # Create language trend DataFrame
        lang_df = pd.DataFrame({
            'date': dates,
            'language': languages
        })
        lang_df.set_index('date', inplace=True)
        
        # Calculate monthly language distribution
        lang_monthly = lang_df.groupby([pd.Grouper(freq='ME'), 'language']).size().unstack(fill_value=0)
        lang_monthly_pct = lang_monthly.div(lang_monthly.sum(axis=1), axis=0) * 100
        
        # Overall statistics
        lang_counts = Counter(languages)
        total_conversations = len(languages)
        
        return {
            "monthly_trends": lang_monthly_pct,
            "overall_distribution": {
                lang: round(count/total_conversations * 100, 1) 
                for lang, count in lang_counts.items()
            },
            "total_conversations": total_conversations,
            "monthly_raw_counts": lang_monthly
        }
    
    def build_topic_timeline(self, discovered_data: Dict[str, Any], freq='D'):
        """Create timeline for discovered topics"""
        organized_topics = discovered_data.get("organized_topics", {})
        conversation_topics = discovered_data.get("conversation_topics", {})
        
        if not organized_topics or not conversation_topics:
            return pd.DataFrame()
        
        # Get canonical mapping to consolidate variations
        canonical_mapping = organized_topics.get("canonical_mapping", {})
        topic_counts = organized_topics.get("topic_counts", {})
        
        # Get top topics for visualization
        top_topics = sorted(topic_counts.keys(), key=lambda x: topic_counts[x], reverse=True)[:10]
        
        timeline_data = {}
        
        for topic in top_topics:
            # Create series for this topic
            topic_series = pd.Series(0, index=self.df.index)
            
            # Find conversations that mention this topic (or its variations)
            for conv_id, topics in conversation_topics.items():
                if conv_id in self.df.index:
                    # Check if any topic in this conversation maps to our target topic
                    for conv_topic in topics:
                        canonical_topic = canonical_mapping.get(conv_topic.lower(), conv_topic.lower())
                        if canonical_topic == topic.lower():
                            topic_series[conv_id] = 1
                            break
            
            # Resample to show trends
            timeline_data[topic] = topic_series.resample(freq).sum()
        
        return pd.DataFrame(timeline_data).fillna(0)
    
    def get_topic_category_data(self, discovered_data: Dict[str, Any]):
        """Prepare category-based topic data for visualization"""
        organized_topics = discovered_data.get("organized_topics", {})
        
        if not organized_topics:
            return {}
        
        categories = organized_topics.get("categories", {})
        topic_counts = organized_topics.get("topic_counts", {})
        
        category_data = {}
        
        for category, topics_in_cat in categories.items():
            category_data[category] = {}
            for topic in topics_in_cat:
                count = topic_counts.get(topic.lower(), 0)
                if count > 0:
                    category_data[category][topic] = count
        
        return category_data

class MetaAnalyzer:
    """Level 2: Генерирует глубокие инсайты с помощью GPT-4o"""
    
    def __init__(self, analytics_engine: AnalyticsEngine):
        self.engine = analytics_engine
        # Оборачиваем клиент для трассировки
        self.client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    async def generate_insights(self) -> Dict[str, Any]:
        print(f"🧠 Generating deep insights using {ANALYSIS_MODEL_L2}...")
        balance_scores = self.engine.get_life_balance_wheel()
        neglected_areas = self.engine.get_neglected_areas()
        top_domains = self.engine.df['domain'].value_counts().head(3).to_dict()
        
        # Add temporal and interest analysis
        temporal = TemporalAnalyzer(self.engine.df)
        interests = InterestTrendAnalyzer(self.engine.df)
        
        daily_pattern = temporal.get_daily_usage_pattern()
        weekly_pattern = temporal.get_weekly_usage_pattern()
        style_evolution = temporal.get_communication_style_evolution()
        all_interests = interests.extract_all_interests()
        top_interests = sorted(all_interests.keys(), key=lambda x: all_interests[x]['count'], reverse=True)[:10]
        
        prompt = f"""
Проанализируй данные об использовании ChatGPT за длительный период. Создай глубокий психологический и поведенческий профиль пользователя на основе всех доступных данных.

**Колесо баланса жизни (средние оценки 0-10):**
{balance_scores.to_string()}

**Критические пробелы (оценка < 2.0):**
{neglected_areas.to_string() if not neglected_areas.empty else "Нет критических пробелов"}

**Топ-3 домена интересов:**
{json.dumps(top_domains, indent=2)}

**Поведенческие паттерны:**
- Пик активности по часам: {daily_pattern.idxmax()}:00 ({daily_pattern.max()} диалогов)
- Самый активный день недели: {weekly_pattern.idxmax()} ({weekly_pattern.max()} диалогов)
- Общее количество диалогов: {len(self.engine.df)}

**Топ-10 обнаруженных интересов:**
{', '.join(top_interests) if top_interests else "Интересы не обнаружены"}

**Эволюция стиля общения (если есть данные):**
{style_evolution.mean().to_string() if not style_evolution.empty else "Недостаточно данных для анализа эволюции"}

ЗАДАЧИ:
1.  **Behavioral Analysis**: Проанализируй поведенческие паттерны и что они говорят о личности.
2.  **Interest Evolution**: Как менялись интересы и что это означает?
3.  **Communication Growth**: Как эволюционировал стиль общения? Какие тренды видны?
4.  **Life Balance Deep Dive**: Глубокий анализ баланса с учетом временных данных.
5.  **Personality & Growth**: Комплексный психологический портрет и признаки роста/изменений.
6.  **Strategic Recommendations**: 3 конкретные рекомендации на основе всех данных.

Ответь в формате JSON:
{{
  "behavioral_analysis": "string",
  "interest_evolution": "string", 
  "communication_growth": "string",
  "life_balance_analysis": "string",
  "personality_sketch": "string",
  "recommendations": ["rec1", "rec2", "rec3"]
}}
"""
        try:
            # o3 model only supports default temperature (1.0)
            if ANALYSIS_MODEL_L2 == "o3":
                response = await self.client.chat.completions.create(
                    model=ANALYSIS_MODEL_L2,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}
                )
            else:
                response = await self.client.chat.completions.create(
                    model=ANALYSIS_MODEL_L2,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.5,
                    response_format={"type": "json_object"}
                )
            insights = json.loads(response.choices[0].message.content)
            print("✅ Deep insights generated.")
            return insights
        except Exception as e:
            print(f"❌ Error generating insights: {e}")
            return {}

class ReportGenerator:
    """Создает итоговый HTML отчет"""
    
    def __init__(self, engine: AnalyticsEngine, insights: Dict[str, Any]):
        self.engine = engine
        self.insights = insights
        self.life_balance_labels = list(ConversationProcessor(None).life_balance_schema.values())
    
    async def create_visuals_and_report(self):
        # Initialize analyzers
        temporal = TemporalAnalyzer(self.engine.df)
        interests = InterestTrendAnalyzer(self.engine.df)
        
        # 1. Life Balance Wheel
        balance_scores = self.engine.get_life_balance_wheel()
        fig_wheel = go.Figure(data=go.Scatterpolar(
            r=balance_scores.values, theta=self.life_balance_labels,
            fill='toself', name='Life Balance'
        ))
        fig_wheel.update_layout(title="🎯 Колесо баланса жизни", template="plotly_dark")
        
        # 2. Balance Evolution - Use appropriate resampling based on time span
        time_span = (self.engine.df.index.max() - self.engine.df.index.min()).days
        if time_span <= 7:
            freq = 'D'  # Daily for week or less
        elif time_span <= 30:
            freq = 'D'  # Daily for month or less  
        else:
            freq = 'ME'  # Monthly for longer periods
            
        evolution_df = self.engine.get_balance_evolution(freq=freq)
        if not evolution_df.empty:
            fig_evolution = px.line(
                evolution_df, x=evolution_df.index, y=evolution_df.columns,
                title="📈 Эволюция баланса во времени", template="plotly_dark",
                labels={'value': 'Оценка (0-10)', 'date': 'Дата', 'variable': 'Сфера жизни'}
            )
        else:
            # Fallback: use raw data points
            balance_over_time = self.engine.df[self.engine.life_balance_keys]
            fig_evolution = px.line(
                balance_over_time, x=balance_over_time.index, y=balance_over_time.columns,
                title="📈 Эволюция баланса во времени (детализированно)", template="plotly_dark",
                labels={'value': 'Оценка (0-10)', 'date': 'Дата', 'variable': 'Сфера жизни'}
            )
        
        # 3. Enhanced Communication Style Evolution - Use appropriate resampling
        comm_df = temporal.get_communication_style_evolution(freq=freq)
        if not comm_df.empty:
            fig_comm = px.line(
                comm_df, x=comm_df.index, y=comm_df.columns,
                title="💬 Эволюция стиля общения (расширенная)", template="plotly_dark",
                labels={'value': 'Оценка (0-10)', 'date': 'Дата', 'variable': 'Стиль'}
            )
        else:
            # Fallback: use raw data points
            style_cols = ['politeness', 'technical_depth', 'curiosity', 'confidence', 'formality']
            comm_over_time = self.engine.df[style_cols]
            fig_comm = px.line(
                comm_over_time, x=comm_over_time.index, y=comm_over_time.columns,
                title="💬 Эволюция стиля общения (детализированно)", template="plotly_dark",
                labels={'value': 'Оценка (0-10)', 'date': 'Дата', 'variable': 'Стиль'}
            )
        
        # 4. Daily Usage Pattern
        daily_usage = temporal.get_daily_usage_pattern()
        fig_daily = px.bar(
            x=daily_usage.index, y=daily_usage.values,
            title="⏰ Паттерн использования по часам дня", template="plotly_dark",
            labels={'x': 'Час дня', 'y': 'Количество диалогов'}
        )
        
        # 5. Weekly Usage Pattern  
        weekly_usage = temporal.get_weekly_usage_pattern()
        fig_weekly = px.bar(
            x=weekly_usage.index, y=weekly_usage.values,
            title="📅 Паттерн использования по дням недели", template="plotly_dark",
            labels={'x': 'День недели', 'y': 'Количество диалогов'}
        )
        
        # 6. Interest Trends - Use appropriate frequency
        interest_trends = interests.get_interest_trends(top_n=8, freq=freq)
        if not interest_trends.empty:
            fig_interests = px.line(
                interest_trends, x=interest_trends.index, y=interest_trends.columns,
                title="🎯 Тренды интересов во времени", template="plotly_dark",
                labels={'value': 'Количество упоминаний', 'date': 'Дата', 'variable': 'Интерес'}
            )
        else:
            fig_interests = go.Figure().add_annotation(text="Недостаточно данных об интересах")
            fig_interests.update_layout(title="🎯 Тренды интересов во времени", template="plotly_dark")
        
        # 7. Session Type Evolution - Use appropriate frequency
        session_trends = temporal.get_session_type_trends(freq=freq)
        if not session_trends.empty:
            fig_sessions = px.area(
                session_trends, x=session_trends.index, y=session_trends.columns,
                title="🎭 Эволюция типов сессий", template="plotly_dark",
                labels={'value': 'Количество', 'date': 'Дата', 'variable': 'Тип сессии'}
            )
        else:
            fig_sessions = go.Figure().add_annotation(text="Недостаточно данных о типах сессий")
            fig_sessions.update_layout(title="🎭 Эволюция типов сессий", template="plotly_dark")
        
        # 8. NEW: LLM-Powered Topic Discovery
        print("🤖 Starting LLM-powered topic discovery...")
        
        # Initialize OpenAI client for topic discovery
        client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Run async topic discovery
        discovered_data = await interests.discover_topics_from_conversations(client)
        
        organized_topics = discovered_data.get("organized_topics", {})
        
        # Get sports topics and language trends
        sports_data = interests.get_sports_topics(discovered_data)
        language_data = interests.get_language_trends()
        
        # 🧠 Personal Evolution Analysis - Deep Transformation Insights
        print("🧠 Starting personal evolution analysis...")
        evolution_analyzer = PersonalEvolutionAnalyzer(self.engine.df)
        sophistication_metrics = evolution_analyzer.calculate_sophistication_metrics()
        evolution_insights = await evolution_analyzer.generate_evolution_insights(sophistication_metrics, client)
        
        if organized_topics and organized_topics.get("topic_counts"):
            # Create topic frequency bar chart
            topic_counts = organized_topics["topic_counts"]
            top_topics = sorted(topic_counts.keys(), key=lambda x: topic_counts[x], reverse=True)[:15]
            top_counts = [topic_counts[topic] for topic in top_topics]
            
            fig_specific_interests = px.bar(
                x=top_topics, y=top_counts,
                title="🎯 AI-Обнаруженные темы и интересы", template="plotly_dark",
                labels={'x': 'Тема/Интерес', 'y': 'Количество упоминаний'},
                color=top_counts,
                color_continuous_scale='viridis'
            )
            fig_specific_interests.update_xaxes(tickangle=45)
            
            # Create topic timeline
            topic_timeline_df = interests.build_topic_timeline(discovered_data, freq=freq)
            if not topic_timeline_df.empty and topic_timeline_df.sum().sum() > 0:
                # Only show topics that have activity
                active_topics = topic_timeline_df.columns[topic_timeline_df.sum() > 0]
                if len(active_topics) > 0:
                    timeline_subset = topic_timeline_df[active_topics]
                    fig_interest_timeline = px.line(
                        timeline_subset, x=timeline_subset.index, y=timeline_subset.columns,
                        title="📈 Временная эволюция AI-обнаруженных тем", template="plotly_dark",
                        labels={'value': 'Активность', 'date': 'Дата', 'variable': 'Тема'}
                    )
                else:
                    fig_interest_timeline = go.Figure().add_annotation(text="Нет временной активности")
                    fig_interest_timeline.update_layout(title="📈 Временная эволюция AI-обнаруженных тем", template="plotly_dark")
            else:
                fig_interest_timeline = go.Figure().add_annotation(text="Недостаточно данных для временного анализа")
                fig_interest_timeline.update_layout(title="📈 Временная эволюция AI-обнаруженных тем", template="plotly_dark")
            
            # Create category visualization
            category_data = interests.get_topic_category_data(discovered_data)
            if category_data:
                # Prepare data for category scatter plot
                scatter_categories = []
                scatter_topics = []
                scatter_counts = []
                
                for category, topics_in_cat in category_data.items():
                    for topic, count in topics_in_cat.items():
                        scatter_categories.append(category)
                        scatter_topics.append(topic)
                        scatter_counts.append(count)
                
                if scatter_counts:
                    fig_interest_heatmap = px.scatter(
                        x=scatter_categories, y=scatter_topics, size=scatter_counts,
                        title="🔥 AI-Организованные темы по категориям", template="plotly_dark",
                        labels={'x': 'Категория', 'y': 'Тема'},
                        size_max=50
                    )
                else:
                    fig_interest_heatmap = go.Figure().add_annotation(text="Нет категорированных тем")
                    fig_interest_heatmap.update_layout(title="🔥 AI-Организованные темы по категориям", template="plotly_dark")
            else:
                fig_interest_heatmap = go.Figure().add_annotation(text="Категории не сформированы")
                fig_interest_heatmap.update_layout(title="🔥 AI-Организованные темы по категориям", template="plotly_dark")
        else:
            # No topics discovered
            fig_specific_interests = go.Figure().add_annotation(text="AI не смог обнаружить четкие темы")
            fig_specific_interests.update_layout(title="🎯 AI-Обнаруженные темы и интересы", template="plotly_dark")
            
            fig_interest_timeline = go.Figure().add_annotation(text="Нет данных для анализа")
            fig_interest_timeline.update_layout(title="📈 Временная эволюция AI-обнаруженных тем", template="plotly_dark")
            
            fig_interest_heatmap = go.Figure().add_annotation(text="Нет данных для анализа")
            fig_interest_heatmap.update_layout(title="🔥 AI-Организованные темы по категориям", template="plotly_dark")

        # 7. Sports Topics Analysis
        sports_topics = sports_data.get("sports_topics", {})
        if sports_topics:
            sport_names = list(sports_topics.keys())
            sport_counts = list(sports_topics.values())
            
            fig_sports = px.bar(
                x=sport_counts,
                y=sport_names,
                orientation='h',
                title="⚽ Топ спортивных тем и активностей",
                labels={"x": "Количество упоминаний", "y": "Спортивные темы"},
                template="plotly_dark",
                color=sport_counts,
                color_continuous_scale="Viridis"
            )
            fig_sports.update_layout(height=600, yaxis={'categoryorder':'total ascending'})
            print(f"🏃‍♂️ Found {len(sports_topics)} sports-related topics, total: {sports_data.get('total_sports_mentions', 0)} mentions")
        else:
            fig_sports = go.Figure().add_annotation(text="Спортивные темы не обнаружены")
            fig_sports.update_layout(title="⚽ Топ спортивных тем и активностей", template="plotly_dark")

        # 8. Language Trends Analysis
        lang_monthly = language_data.get("monthly_trends")
        if lang_monthly is not None and not lang_monthly.empty:
            # Create stacked area chart for language trends
            fig_language = go.Figure()
            
            colors = {"russian": "#FF6B6B", "english": "#4ECDC4", "mixed": "#45B7D1", "unknown": "#96CEB4"}
            
            for lang in lang_monthly.columns:
                if lang in colors:
                    fig_language.add_trace(go.Scatter(
                        x=lang_monthly.index,
                        y=lang_monthly[lang],
                        mode='lines',
                        stackgroup='one',
                        name=lang.title(),
                        line_color=colors[lang],
                        fill='tonexty' if lang != lang_monthly.columns[0] else 'tozeroy'
                    ))
            
            fig_language.update_layout(
                title="🌍 Эволюция языков в разговорах",
                xaxis_title="Время",
                yaxis_title="Процент разговоров (%)",
                template="plotly_dark",
                hovermode="x unified",
                height=500
            )
            
            # Add overall distribution info
            overall_dist = language_data.get("overall_distribution", {})
            dist_text = ", ".join([f"{lang}: {pct}%" for lang, pct in overall_dist.items()])
            print(f"🗣️ Language distribution: {dist_text}")
        else:
            fig_language = go.Figure().add_annotation(text="Данные о языках недоступны")
            fig_language.update_layout(title="🌍 Эволюция языков в разговорах", template="plotly_dark")

        # 9. Personal Evolution Dashboard - Then vs Now Analysis
        early_metrics = sophistication_metrics.get("early", {})
        recent_metrics = sophistication_metrics.get("recent", {})
        evolution_scores = sophistication_metrics.get("evolution", {})
        
        if early_metrics and recent_metrics:
            # Create comparative metrics visualization
            metrics_to_compare = [
                ('avg_question_length', 'Question Length'),
                ('vocabulary_richness', 'Vocabulary Richness'),
                ('complexity_ratio', 'Problem Complexity'),
                ('simplicity_ratio', 'Problem Simplicity'),
                ('confidence_ratio', 'Confidence Level'),
                ('uncertainty_ratio', 'Uncertainty Level'),
                ('strategic_focus_ratio', 'Strategic Focus'),
                ('immediate_focus_ratio', 'Immediate Focus'),
                ('avg_curiosity', 'Curiosity'),
                ('avg_technical_depth', 'Technical Depth')
            ]
            
            # Filter available metrics
            available_metrics = [(key, name) for key, name in metrics_to_compare 
                               if key in early_metrics and key in recent_metrics]
            
            if available_metrics:
                metric_names = [name for _, name in available_metrics]
                early_values = [early_metrics[key] for key, _ in available_metrics]
                recent_values = [recent_metrics[key] for key, _ in available_metrics]
                
                fig_evolution = go.Figure()
                
                # Add early period bars
                fig_evolution.add_trace(go.Bar(
                    name='Early Period (First 6 months)',
                    x=metric_names,
                    y=early_values,
                    marker_color='#FF6B6B',
                    opacity=0.8
                ))
                
                # Add recent period bars
                fig_evolution.add_trace(go.Bar(
                    name='Recent Period (Last 6 months)',
                    x=metric_names,
                    y=recent_values,
                    marker_color='#4ECDC4',
                    opacity=0.8
                ))
                
                fig_evolution.update_layout(
                    title="🧠 Personal Evolution: Then vs Now",
                    xaxis_title="Sophistication Metrics",
                    yaxis_title="Score",
                    template="plotly_dark",
                    barmode='group',
                    height=600,
                    showlegend=True
                )
                
                # Add change annotations
                for i, (key, name) in enumerate(available_metrics):
                    if f"{key}_change" in evolution_scores:
                        change = evolution_scores[f"{key}_change"]
                        arrow_color = "#4CAF50" if change > 0 else "#F44336"
                        fig_evolution.add_annotation(
                            x=i,
                            y=max(early_values[i], recent_values[i]) * 1.1,
                            text=f"{change:+.1f}%",
                            showarrow=True,
                            arrowcolor=arrow_color,
                            arrowhead=2,
                            font=dict(color=arrow_color, size=12)
                        )
                
                print(f"🎯 Evolution analysis complete:")
                print(f"  - Early period: {len(evolution_analyzer.early_period)} conversations")
                print(f"  - Recent period: {len(evolution_analyzer.recent_period)} conversations")
                print(f"  - Tracked {len(available_metrics)} sophistication metrics")
                
                # Show biggest changes
                biggest_changes = sorted(
                    [(k.replace('_change', ''), v) for k, v in evolution_scores.items() if k.endswith('_change')],
                    key=lambda x: abs(x[1]), reverse=True
                )[:3]
                
                if biggest_changes:
                    print(f"  - Biggest changes: {', '.join([f'{metric}({change:+.1f}%)' for metric, change in biggest_changes])}")
            else:
                fig_evolution = go.Figure().add_annotation(text="Недостаточно данных для сравнения периодов")
                fig_evolution.update_layout(title="🧠 Personal Evolution: Then vs Now", template="plotly_dark")
        else:
            fig_evolution = go.Figure().add_annotation(text="Недостаточно данных для анализа эволюции")
            fig_evolution.update_layout(title="🧠 Personal Evolution: Then vs Now", template="plotly_dark")

        # 10. MBTI Personality Patterns Analysis
        print("🧠 Creating MBTI personality patterns visualization...")
        
        # Get personality data
        mbti_averages = self.engine.get_mbti_dimensions_averages()
        personality_type_dist = self.engine.get_personality_type_distribution()
        
        if not mbti_averages.empty:
            # Create radar chart for MBTI dimensions
            fig_personality = go.Figure()
            
            # Group opposing dimensions
            dimension_pairs = [
                ('extraversion_score', 'introversion_score', 'E/I'),
                ('sensing_score', 'intuition_score', 'S/N'),
                ('thinking_score', 'feeling_score', 'T/F'),
                ('judging_score', 'perceiving_score', 'J/P')
            ]
            
            dimensions = []
            values = []
            
            for dim1, dim2, label in dimension_pairs:
                if dim1 in mbti_averages and dim2 in mbti_averages:
                    # Calculate preference (positive = first dimension, negative = second dimension)
                    preference = mbti_averages[dim1] - mbti_averages[dim2]
                    dimensions.append(label)
                    values.append(preference)
            
            if values:
                fig_personality = go.Figure(data=go.Scatterpolar(
                    r=[abs(v) for v in values],
                    theta=dimensions,
                    fill='toself',
                    name='MBTI Profile',
                    line_color='#4ECDC4',
                    fillcolor='rgba(78, 205, 196, 0.3)'
                ))
                
                fig_personality.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 5]
                        )
                    ),
                    title="🧠 MBTI Personality Profile",
                    template="plotly_dark",
                    showlegend=False,
                    height=600
                )
                
                # Add type distribution as text
                if not personality_type_dist.empty:
                    top_type = personality_type_dist.index[0]
                    top_count = personality_type_dist.iloc[0]
                    fig_personality.add_annotation(
                        x=0.5, y=0.1,
                        text=f"Most Common Type: {top_type} ({top_count} conversations)",
                        showarrow=False,
                        xref="paper", yref="paper",
                        font=dict(size=14, color="#FFC107")
                    )
            else:
                fig_personality = go.Figure().add_annotation(text="Недостаточно данных для MBTI анализа")
                fig_personality.update_layout(title="🧠 MBTI Personality Profile", template="plotly_dark")
        else:
            fig_personality = go.Figure().add_annotation(text="Данные MBTI не найдены")
            fig_personality.update_layout(title="🧠 MBTI Personality Profile", template="plotly_dark")

        # 11. Decision-Making Patterns Analysis
        print("🎯 Creating decision-making patterns visualization...")
        
        decision_averages = self.engine.get_decision_patterns_averages()
        decision_style_dist = self.engine.get_decision_style_distribution()
        
        if not decision_averages.empty:
            # Create bar chart for decision patterns
            decision_labels = [
                'Fact Gathering', 'Opportunity Seeking', 'Logical Analysis', 
                'Human Factor', 'Planning', 'Improvisation'
            ]
            decision_values = [
                decision_averages.get('fact_gathering_ratio', 0),
                decision_averages.get('opportunity_seeking_ratio', 0),
                decision_averages.get('logical_analysis_ratio', 0),
                decision_averages.get('human_factor_ratio', 0),
                decision_averages.get('planning_ratio', 0),
                decision_averages.get('improvisation_ratio', 0)
            ]
            
            fig_decisions = go.Figure(data=[
                go.Bar(
                    x=decision_labels,
                    y=decision_values,
                    marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3'],
                    text=[f'{v:.1f}' for v in decision_values],
                    textposition='auto'
                )
            ])
            
            fig_decisions.update_layout(
                title="🎯 Decision-Making Patterns",
                xaxis_title="Decision Pattern",
                yaxis_title="Average Score (0-10)",
                template="plotly_dark",
                height=600,
                showlegend=False
            )
            
            # Add dominant decision style info
            if not decision_style_dist.empty:
                top_style = decision_style_dist.index[0]
                top_count = decision_style_dist.iloc[0]
                fig_decisions.add_annotation(
                    x=0.5, y=0.95,
                    text=f"Dominant Style: {top_style} ({top_count} conversations)",
                    showarrow=False,
                    xref="paper", yref="paper",
                    font=dict(size=14, color="#FFC107")
                )
        else:
            fig_decisions = go.Figure().add_annotation(text="Данные о решениях не найдены")
            fig_decisions.update_layout(title="🎯 Decision-Making Patterns", template="plotly_dark")

        # Save all visualizations (existing + new)
        pyo.plot(fig_wheel, filename="life_balance_wheel.html", auto_open=False)
        pyo.plot(fig_evolution, filename="balance_evolution.html", auto_open=False)
        pyo.plot(fig_comm, filename="communication_style.html", auto_open=False)
        pyo.plot(fig_daily, filename="daily_usage_pattern.html", auto_open=False)
        pyo.plot(fig_weekly, filename="weekly_usage_pattern.html", auto_open=False)
        pyo.plot(fig_interests, filename="interest_trends.html", auto_open=False)
        pyo.plot(fig_sessions, filename="session_types.html", auto_open=False)
        
        # Save new interest visualizations
        pyo.plot(fig_specific_interests, filename="specific_interests.html", auto_open=False)
        pyo.plot(fig_interest_timeline, filename="interest_timeline.html", auto_open=False)
        pyo.plot(fig_interest_heatmap, filename="interest_heatmap.html", auto_open=False)
        
        # Save new sports and language visualizations
        pyo.plot(fig_sports, filename="sports_topics.html", auto_open=False)
        pyo.plot(fig_language, filename="language_trends.html", auto_open=False)
        
        # Save evolution analysis visualization
        pyo.plot(fig_evolution, filename="personal_evolution.html", auto_open=False)
        
        # Save new personality visualizations
        pyo.plot(fig_personality, filename="mbti_personality_profile.html", auto_open=False)
        pyo.plot(fig_decisions, filename="decision_making_patterns.html", auto_open=False)
        
        print("✅ Visualizations saved to .html files (including MBTI analysis and decision patterns)")
        
        # Generate summary stats for the report
        temporal = TemporalAnalyzer(self.engine.df)
        interests = InterestTrendAnalyzer(self.engine.df)
        all_interests = interests.extract_all_interests()
        top_interests = sorted(all_interests.keys(), key=lambda x: all_interests[x]['count'], reverse=True)[:5]
        
        # Get LLM-discovered topics for summary
        if organized_topics and organized_topics.get("topic_counts"):
            topic_counts = organized_topics["topic_counts"]
            top_specific_interests = sorted(topic_counts.keys(), key=lambda x: topic_counts[x], reverse=True)[:5]
        else:
            top_specific_interests = []
        
        # Get personality summary
        personality_type_dist = self.engine.get_personality_type_distribution()
        decision_style_dist = self.engine.get_decision_style_distribution()
        
        top_personality_type = personality_type_dist.index[0] if not personality_type_dist.empty else "Unknown"
        top_decision_style = decision_style_dist.index[0] if not decision_style_dist.empty else "Unknown"
        
        daily_peak = temporal.get_daily_usage_pattern().idxmax()
        weekly_peak = temporal.get_weekly_usage_pattern().idxmax()
        
        # Determine analysis granularity based on time span
        time_span = (self.engine.df.index.max() - self.engine.df.index.min()).days
        if time_span <= 7:
            analysis_mode = "daily_trends"
            granularity = "ежедневные тренды"
        elif time_span <= 30:
            analysis_mode = "daily_trends" 
            granularity = "ежедневные тренды"
        else:
            analysis_mode = "monthly_trends"
            granularity = "месячные тренды"
        
        # HTML Report
        report_html = f"""
        <html><head><title>Structured Analysis V3 - Enhanced</title>
        <style> 
            body {{ font-family: sans-serif; background-color: #111; color: #eee; margin: 40px; }}
            h1 {{ color: #4CAF50; }}
            h2 {{ color: #FFC107; border-bottom: 1px solid #333; }}
            .stat-box {{ background: #222; padding: 15px; margin: 10px 0; border-radius: 5px; }}
            .link-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 10px; }}
            .link-card {{ background: #333; padding: 15px; border-radius: 5px; text-align: center; }}
            .link-card a {{ color: #4CAF50; text-decoration: none; font-weight: bold; }}
        </style>
        </head><body>
            <h1>🚀 Интеллектуальный анализ V3 - Расширенный</h1>
            
            <div class="stat-box">
                <h2>📊 Быстрая статистика</h2>
                <p><strong>Всего диалогов:</strong> {len(self.engine.df)}</p>
                <p><strong>Период анализа:</strong> {self.engine.df.index.min().strftime('%Y-%m-%d')} - {self.engine.df.index.max().strftime('%Y-%m-%d')} ({time_span} дней)</p>
                <p><strong>Режим анализа:</strong> 📈 Тренды ({granularity})</p>
                <p><strong>Пик активности:</strong> {daily_peak}:00 в {weekly_peak}</p>
                <p><strong>Топ-5 интересов (AI):</strong> {', '.join(top_interests[:5]) if top_interests else 'Не обнаружено'}</p>
                <p><strong>AI-обнаруженные темы:</strong> {', '.join(top_specific_interests[:5]) if top_specific_interests else 'Анализ не завершен'}</p>
                <p><strong>Доминирующий MBTI тип:</strong> {top_personality_type}</p>
                <p><strong>Стиль принятия решений:</strong> {top_decision_style}</p>
            </div>
            
            <h2>🧠 AI-инсайты</h2>
            <div class="stat-box">
                <h3>🕐 Поведенческий анализ:</h3><p>{self.insights.get('behavioral_analysis', 'Анализ недоступен')}</p>
                <h3>🎯 Эволюция интересов:</h3><p>{self.insights.get('interest_evolution', 'Анализ недоступен')}</p>
                <h3>💬 Развитие коммуникации:</h3><p>{self.insights.get('communication_growth', 'Анализ недоступен')}</p>
                <h3>⚖️ Анализ баланса:</h3><p>{self.insights.get('life_balance_analysis', 'Анализ недоступен')}</p>
                <h3>👤 Психологический портрет:</h3><p>{self.insights.get('personality_sketch', 'Анализ недоступен')}</p>
                <h3>🎯 Рекомендации:</h3><ul>{''.join(f'<li>{rec}</li>' for rec in self.insights.get('recommendations', ['Рекомендации недоступны']))}</ul>
            </div>
            
            <h2>🧬 Анализ личностной эволюции</h2>
            <div class="stat-box">
                <h3>🧠 Когнитивное развитие:</h3>
                <p><strong>Сложность мышления:</strong> {evolution_insights.get('cognitive_evolution', {}).get('thinking_complexity', 'Анализ недоступен')}</p>
                <p><strong>Подход к решению проблем:</strong> {evolution_insights.get('cognitive_evolution', {}).get('problem_solving_approach', 'Анализ недоступен')}</p>
                <p><strong>Стиль обучения:</strong> {evolution_insights.get('cognitive_evolution', {}).get('learning_style', 'Анализ недоступен')}</p>
                
                <h3>👤 Развитие личности:</h3>
                <p><strong>Рост уверенности:</strong> {evolution_insights.get('personality_development', {}).get('confidence_growth', 'Анализ недоступен')}</p>
                <p><strong>Интеллектуальное любопытство:</strong> {evolution_insights.get('personality_development', {}).get('intellectual_curiosity', 'Анализ недоступен')}</p>
                <p><strong>Стиль коммуникации:</strong> {evolution_insights.get('personality_development', {}).get('communication_style', 'Анализ недоступен')}</p>
                
                <h3>🎯 Эволюция жизненного фокуса:</h3>
                <p><strong>Смещение приоритетов:</strong> {evolution_insights.get('life_focus_evolution', {}).get('priority_shifts', 'Анализ недоступен')}</p>
                <p><strong>Расширение кругозора:</strong> {evolution_insights.get('life_focus_evolution', {}).get('scope_expansion', 'Анализ недоступен')}</p>
                <p><strong>Временные горизонты:</strong> {evolution_insights.get('life_focus_evolution', {}).get('time_horizon', 'Анализ недоступен')}</p>
                
                <h3>📈 Траектория роста:</h3>
                <p><strong>Ключевые изменения:</strong> {evolution_insights.get('growth_trajectory', {}).get('biggest_changes', 'Анализ недоступен')}</p>
                <p><strong>Области развития:</strong> {evolution_insights.get('growth_trajectory', {}).get('growth_areas', 'Анализ недоступен')}</p>
                <p><strong>Будущее направление:</strong> {evolution_insights.get('growth_trajectory', {}).get('future_direction', 'Анализ недоступен')}</p>
            </div>
            
            <h2>📈 Интерактивные визуализации</h2>
            <div class="link-grid">
                <div class="link-card">
                    <a href="life_balance_wheel.html">🎯 Колесо баланса жизни</a>
                    <p>Ваши сферы жизни</p>
                </div>
                <div class="link-card">
                    <a href="balance_evolution.html">📈 Эволюция баланса</a>
                    <p>Изменения во времени</p>
                </div>
                <div class="link-card">
                    <a href="communication_style.html">💬 Стиль общения</a>
                    <p>Эволюция стиля</p>
                </div>
                <div class="link-card">
                    <a href="daily_usage_pattern.html">⏰ Дневные паттерны</a>
                    <p>Когда вы наиболее активны</p>
                </div>
                <div class="link-card">
                    <a href="weekly_usage_pattern.html">📅 Недельные паттерны</a>
                    <p>Ваши предпочтения по дням недели</p>
                </div>
                <div class="link-card">
                    <a href="interest_trends.html">🎯 Тренды интересов</a>
                    <p>Эволюция ваших увлечений</p>
                </div>
                <div class="link-card">
                    <a href="session_types.html">🎭 Эволюция сессий</a>
                    <p>Как менялись ваши запросы</p>
                </div>
                <div class="link-card">
                    <a href="personal_evolution.html">🧬 Личностная эволюция</a>
                    <p>Тогда vs Сейчас: Ваше развитие</p>
                </div>
            </div>
            
            <h2>🧠 16 Personalities & Decision-Making Analysis</h2>
            <div class="link-grid">
                <div class="link-card">
                    <a href="mbti_personality_profile.html">🧠 MBTI Personality Profile</a>
                    <p>Ваш профиль 16 personalities на основе паттернов общения</p>
                </div>
                <div class="link-card">
                    <a href="decision_making_patterns.html">🎯 Decision-Making Patterns</a>
                    <p>Анализ ваших паттернов принятия решений</p>
                </div>
            </div>
            
            <h2>🤖 AI-Анализ тем и интересов</h2>
            <div class="link-grid">
                <div class="link-card">
                    <a href="specific_interests.html">🎯 AI-Обнаруженные темы</a>
                    <p>Интеллектуальное извлечение ваших интересов</p>
                </div>
                <div class="link-card">
                    <a href="interest_timeline.html">📈 Эволюция тем во времени</a>
                    <p>Как менялись ваши интересы</p>
                </div>
                <div class="link-card">
                    <a href="interest_heatmap.html">🔥 Категории тем</a>
                    <p>AI-организация по смысловым группам</p>
                </div>
            </div>
        </body></html>"""
        
        with open("structured_report_v3.html", "w", encoding='utf-8') as f:
            f.write(report_html)
        print("✅ Final report created: structured_report_v3.html")


async def main():
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY not found.")
        return
    
    # Clear previous results file if it exists
    if os.path.exists(RESULTS_FILE):
        os.remove(RESULTS_FILE)
        print(f"🗑️ Removed previous results file: {RESULTS_FILE}")
    
    # --- Level 1 ---
    print("--- Starting Phase 1: Analyzing conversations ---")
    pipeline = PipelineManager('conversations.json.zip')
    await pipeline.run_analysis_pipeline()
    
    if not os.path.exists(RESULTS_FILE):
        print(f"❌ File {RESULTS_FILE} was not created. Exiting.")
        return
    
    # --- Level 2 ---
    print("\n--- Starting Phase 2: Aggregation and insights generation ---")
    try:
        engine = AnalyticsEngine(RESULTS_FILE)
        meta_analyzer = MetaAnalyzer(engine)
        insights = await meta_analyzer.generate_insights()
        
        if insights:
            report_gen = ReportGenerator(engine, insights)
            await report_gen.create_visuals_and_report()
            print(f"\n🎉 ALL DONE! Open structured_report_v3.html")
        else:
            print("⚠️ Failed to generate insights, but data aggregated.")
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ Error processing results: {e}")
        print("💡 Try running the analysis again or check OpenAI models.")

if __name__ == "__main__":
    asyncio.run(main())
