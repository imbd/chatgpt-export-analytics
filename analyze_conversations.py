#!/usr/bin/env python3
"""
ChatGPT Conversations Analysis Tool
Creates multi-level visual analytics of ChatGPT export data
"""

import json
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from datetime import datetime, timedelta
import re
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# Style setup
plt.style.use('dark_background')
sns.set_palette("husl")

class ChatGPTAnalyzer:
    def __init__(self, zip_path):
        self.zip_path = zip_path
        self.conversations = []
        self.df = None
        
    def load_data(self):
        """Loads data from ZIP archive"""
        print("📁 Loading data from archive...")
        
        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            with zip_ref.open('conversations.json') as f:
                self.conversations = json.load(f)
        
        print(f"✅ Loaded {len(self.conversations)} conversations")
        return self
    
    def prepare_dataframe(self):
        """Prepares DataFrame for analysis"""
        print("🔄 Preparing data for analysis...")
        
        data = []
        for conv in self.conversations:
            # Basic information
            conv_data = {
                'id': conv.get('id', ''),
                'title': conv.get('title', 'Untitled'),
                'create_time': conv.get('create_time'),
                'update_time': conv.get('update_time'),
                'model': conv.get('default_model_slug', 'unknown'),
                'is_archived': conv.get('is_archived', False),
                'is_starred': conv.get('is_starred', False),
                'gizmo_id': conv.get('gizmo_id'),
                'conversation_origin': conv.get('conversation_origin'),
            }
            
            # Analyze mapping for message counting
            mapping = conv.get('mapping', {})
            
            user_messages = 0
            assistant_messages = 0
            system_messages = 0
            tool_messages = 0
            total_chars = 0
            languages = set()
            
            for node_id, node in mapping.items():
                if node and node.get('message') and node['message'].get('content') and node['message'].get('author'):
                    role = node['message']['author'].get('role', 'unknown')
                    content = node['message']['content']
                    
                    if content.get('content_type') == 'text' and content.get('parts'):
                        text = ' '.join(content['parts']).strip()
                        total_chars += len(text)
                        
                        # Language detection
                        if re.search(r'[а-яё]', text, re.IGNORECASE):
                            languages.add('ru')
                        if re.search(r'[a-z]', text, re.IGNORECASE):
                            languages.add('en')
                        if re.search(r'[ąćęłńóśźż]', text, re.IGNORECASE):
                            languages.add('pl')
                        
                        # Count by roles
                        if role == 'user':
                            user_messages += 1
                        elif role == 'assistant':
                            assistant_messages += 1
                        elif role == 'system':
                            system_messages += 1
                        elif role == 'tool':
                            tool_messages += 1
            
            conv_data.update({
                'total_nodes': len(mapping),
                'user_messages': user_messages,
                'assistant_messages': assistant_messages,
                'system_messages': system_messages,
                'tool_messages': tool_messages,
                'total_messages': user_messages + assistant_messages + system_messages + tool_messages,
                'total_chars': total_chars,
                'avg_chars_per_message': total_chars / max(1, user_messages + assistant_messages),
                'languages': list(languages),
                'primary_language': self._detect_primary_language(conv_data['title'], languages),
                'is_gpt': bool(conv.get('gizmo_id')),
                'conversation_complexity': self._calculate_complexity(mapping),
            })
            
            # Time metrics
            if conv_data['create_time'] and conv_data['update_time']:
                duration = conv_data['update_time'] - conv_data['create_time']
                conv_data['duration_minutes'] = duration / 60
                conv_data['create_datetime'] = datetime.fromtimestamp(conv_data['create_time'])
                conv_data['update_datetime'] = datetime.fromtimestamp(conv_data['update_time'])
                conv_data['hour_of_day'] = conv_data['create_datetime'].hour
                conv_data['day_of_week'] = conv_data['create_datetime'].weekday()
                conv_data['month'] = conv_data['create_datetime'].month
                conv_data['year'] = conv_data['create_datetime'].year
            
            data.append(conv_data)
        
        self.df = pd.DataFrame(data)
        print(f"✅ Prepared DataFrame with {len(self.df)} records and {len(self.df.columns)} features")
        return self
    
    def _detect_primary_language(self, title, content_languages):
        """Detects primary language of the conversation"""
        if re.search(r'[а-яё]', title, re.IGNORECASE):
            return 'ru'
        elif 'ru' in content_languages and len(content_languages) == 1:
            return 'ru'
        elif 'en' in content_languages:
            return 'en'
        elif 'pl' in content_languages:
            return 'pl'
        else:
            return 'unknown'
    
    def _calculate_complexity(self, mapping):
        """Calculates conversation complexity based on structure"""
        if not mapping:
            return 0
        
        # Complexity factors:
        # - Number of nodes
        # - Branching depth
        # - Tool usage
        # - Message length
        
        node_count = len(mapping)
        tool_usage = 0
        
        for node in mapping.values():
            if node and node.get('message') and node['message'].get('author'):
                if node['message']['author'].get('role') == 'tool':
                    tool_usage += 1
        
        # Simple complexity metric
        complexity = min(10, (node_count / 10) + (tool_usage * 2))
        return round(complexity, 1)
    
    def create_time_series_analysis(self):
        """Creates time series analysis"""
        print("📊 Creating time series analysis...")
        
        # Data preparation
        df_time = self.df[self.df['create_datetime'].notna()].copy()
        df_time['date'] = df_time['create_datetime'].dt.date
        
        # Group by days
        daily_stats = df_time.groupby('date').agg({
            'id': 'count',
            'total_messages': 'sum',
            'total_chars': 'sum',
            'model': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'unknown'
        }).rename(columns={'id': 'conversations_count'})
        
        # Create charts
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Conversations by Day', 'Cumulative Statistics',
                'Activity by Hour', 'Activity by Day of Week',
                'Model Evolution', 'Language Distribution over Time'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": True}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Conversations by day
        fig.add_trace(
            go.Scatter(
                x=daily_stats.index,
                y=daily_stats['conversations_count'],
                mode='lines+markers',
                name='Conversations/day',
                line=dict(color='#00ff88', width=2)
            ),
            row=1, col=1
        )
        
        # 2. Cumulative statistics
        cumulative_convs = daily_stats['conversations_count'].cumsum()
        cumulative_messages = daily_stats['total_messages'].cumsum()
        
        fig.add_trace(
            go.Scatter(
                x=daily_stats.index,
                y=cumulative_convs,
                mode='lines',
                name='Total conversations',
                line=dict(color='#ff6b35', width=2)
            ),
            row=1, col=2
        )
        
        # 3. Activity by hour
        hourly_activity = df_time.groupby('hour_of_day').size()
        fig.add_trace(
            go.Bar(
                x=hourly_activity.index,
                y=hourly_activity.values,
                name='By hours',
                marker_color='#4ecdc4'
            ),
            row=2, col=1
        )
        
        # 4. Activity by day of week
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        weekly_activity = df_time.groupby('day_of_week').size()
        fig.add_trace(
            go.Bar(
                x=[days[i] for i in weekly_activity.index],
                y=weekly_activity.values,
                name='By days',
                marker_color='#ff9999'
            ),
            row=2, col=2
        )
        
        # 5. Model evolution over time
        model_evolution = df_time.groupby(['date', 'model']).size().unstack(fill_value=0)
        for model in model_evolution.columns[:5]:  # Top 5 models
            fig.add_trace(
                go.Scatter(
                    x=model_evolution.index,
                    y=model_evolution[model],
                    mode='lines',
                    name=model,
                    stackgroup='one'
                ),
                row=3, col=1
            )
        
        # 6. Languages over time
        lang_by_time = df_time.groupby(['date', 'primary_language']).size().unstack(fill_value=0)
        for lang in lang_by_time.columns:
            fig.add_trace(
                go.Scatter(
                    x=lang_by_time.index,
                    y=lang_by_time[lang],
                    mode='lines',
                    name=f'Language: {lang}',
                    stackgroup='two'
                ),
                row=3, col=2
            )
        
        fig.update_layout(
            height=1200,
            title_text="📈 Temporal Analysis of ChatGPT Usage",
            title_x=0.5,
            showlegend=True,
            template='plotly_dark'
        )
        
        return fig
    
    def create_model_analysis(self):
        """Model usage analysis"""
        print("🤖 Analyzing model usage...")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Model Distribution',
                'Conversation Complexity by Model',
                'Duration by Model',
                'Model Efficiency'
            ],
            specs=[[{"type": "domain"}, {"type": "violin"}],
                   [{"type": "box"}, {"type": "scatter"}]]
        )
        
        # 1. Models pie chart
        model_counts = self.df['model'].value_counts()
        fig.add_trace(
            go.Pie(
                labels=model_counts.index[:8],
                values=model_counts.values[:8],
                name="Models",
                hole=0.4
            ),
            row=1, col=1
        )
        
        # 2. Complexity violin plot
        top_models = model_counts.head(5).index
        for model in top_models:
            model_data = self.df[self.df['model'] == model]['conversation_complexity']
            fig.add_trace(
                go.Violin(
                    y=model_data,
                    name=model,
                    box_visible=True,
                    meanline_visible=True
                ),
                row=1, col=2
            )
        
        # 3. Duration box plot
        duration_data = self.df[self.df['duration_minutes'].notna()]
        for model in top_models:
            model_duration = duration_data[duration_data['model'] == model]['duration_minutes']
            if len(model_duration) > 0:
                fig.add_trace(
                    go.Box(
                        y=model_duration,
                        name=model,
                        boxpoints='outliers'
                    ),
                    row=2, col=1
                )
        
        # 4. Efficiency scatter plot (messages vs time)
        efficiency_data = self.df[
            (self.df['duration_minutes'].notna()) & 
            (self.df['duration_minutes'] > 0) & 
            (self.df['total_messages'] > 0)
        ].copy()
        
        efficiency_data['messages_per_minute'] = efficiency_data['total_messages'] / efficiency_data['duration_minutes']
        
        for model in top_models:
            model_eff = efficiency_data[efficiency_data['model'] == model]
            fig.add_trace(
                go.Scatter(
                    x=model_eff['total_messages'],
                    y=model_eff['messages_per_minute'],
                    mode='markers',
                    name=model,
                    opacity=0.7
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            title_text="🤖 ChatGPT Model Analysis",
            title_x=0.5,
            showlegend=True,
            template='plotly_dark'
        )
        
        return fig
    
    def create_usage_patterns(self):
        """Usage patterns analysis"""
        print("🔍 Analyzing usage patterns...")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Activity Heatmap',
                'Conversation Complexity Distribution',
                'Feature Correlation',
                'Usage Type Clusters'
            ]
        )
        
        # 1. Activity heatmap by hours and days
        df_time = self.df[self.df['create_datetime'].notna()].copy()
        if len(df_time) > 0:
            activity_matrix = df_time.groupby(['day_of_week', 'hour_of_day']).size().unstack(fill_value=0)
            
            fig.add_trace(
                go.Heatmap(
                    z=activity_matrix.values,
                    x=list(range(24)),
                    y=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                    colorscale='Viridis',
                    name='Activity'
                ),
                row=1, col=1
            )
        
        # 2. Complexity histogram
        fig.add_trace(
            go.Histogram(
                x=self.df['conversation_complexity'],
                nbinsx=20,
                name='Complexity',
                marker_color='#ff6b35'
            ),
            row=1, col=2
        )
        
        # 3. Correlation matrix
        numeric_cols = ['total_messages', 'total_chars', 'conversation_complexity', 'duration_minutes']
        corr_data = self.df[numeric_cols].corr()
        
        fig.add_trace(
            go.Heatmap(
                z=corr_data.values,
                x=corr_data.columns,
                y=corr_data.columns,
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr_data.values, 2),
                texttemplate="%{text}",
                textfont={"size": 10}
            ),
            row=2, col=1
        )
        
        # 4. Scatter plot for clustering
        fig.add_trace(
            go.Scatter(
                x=self.df['total_messages'],
                y=self.df['conversation_complexity'],
                mode='markers',
                marker=dict(
                    size=self.df['total_chars'] / 1000,
                    color=self.df['duration_minutes'],
                    colorscale='Plasma',
                    showscale=True,
                    sizemode='diameter',
                    sizeref=2.*max(self.df['total_chars']/1000)/(40.**2),
                    sizemin=4
                ),
                name='Conversations',
                text=self.df['title'],
                hovertemplate='<b>%{text}</b><br>Messages: %{x}<br>Complexity: %{y}<extra></extra>'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text="🔍 ChatGPT Usage Patterns",
            title_x=0.5,
            showlegend=True,
            template='plotly_dark'
        )
        
        return fig
    
    def create_content_analysis(self):
        """Content and topics analysis"""
        print("📝 Analyzing content and topics...")
        
        # Title analysis
        all_titles = ' '.join(self.df['title'].fillna('').str.lower())
        
        # Keyword extraction
        russian_words = re.findall(r'\b[а-яё]{3,}\b', all_titles)
        english_words = re.findall(r'\b[a-z]{3,}\b', all_titles)
        
        ru_counter = Counter(russian_words)
        en_counter = Counter(english_words)
        
        # Create word cloud through bar charts
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Top Russian Words in Titles',
                'Top English Words in Titles',
                'Language Distribution',
                'Topic Dynamics over Time'
            ],
            specs=[[{"type": "xy"}, {"type": "xy"}],
                   [{"type": "domain"}, {"type": "xy"}]]
        )
        
        # 1. Russian words
        ru_top = ru_counter.most_common(15)
        if ru_top:
            fig.add_trace(
                go.Bar(
                    x=[word for word, count in ru_top],
                    y=[count for word, count in ru_top],
                    name='Russian words',
                    marker_color='#ff6b6b'
                ),
                row=1, col=1
            )
        
        # 2. English words
        en_top = en_counter.most_common(15)
        if en_top:
            fig.add_trace(
                go.Bar(
                    x=[word for word, count in en_top],
                    y=[count for word, count in en_top],
                    name='English words',
                    marker_color='#4ecdc4'
                ),
                row=1, col=2
            )
        
        # 3. Language distribution
        lang_dist = self.df['primary_language'].value_counts()
        fig.add_trace(
            go.Pie(
                labels=lang_dist.index,
                values=lang_dist.values,
                name="Languages"
            ),
            row=2, col=1
        )
        
        # 4. Popular topics dynamics
        df_time = self.df[self.df['create_datetime'].notna()].copy()
        if len(df_time) > 0:
            # Search for trending words
            trending_words = ['ai', 'code', 'python', 'analysis', 'data', 'agent', 'api']
            df_time['month_year'] = df_time['create_datetime'].dt.to_period('M')
            
            for word in trending_words:
                word_counts = []
                months = []
                for period in df_time['month_year'].unique():
                    month_titles = df_time[df_time['month_year'] == period]['title'].str.lower().str.cat(sep=' ')
                    count = len(re.findall(rf'\b{word}\b', month_titles))
                    if count > 0:
                        word_counts.append(count)
                        months.append(str(period))
                
                if word_counts:
                    fig.add_trace(
                        go.Scatter(
                            x=months,
                            y=word_counts,
                            mode='lines+markers',
                            name=word
                        ),
                        row=2, col=2
                    )
        
        fig.update_layout(
            height=800,
            title_text="📝 Content and Topics Analysis",
            title_x=0.5,
            showlegend=True,
            template='plotly_dark'
        )
        
        return fig
    
    def generate_insights(self):
        """Generates key insights"""
        print("💡 Generating insights...")
        
        insights = []
        
        # Basic statistics
        total_conversations = len(self.df)
        total_messages = self.df['total_messages'].sum()
        total_chars = self.df['total_chars'].sum()
        avg_complexity = self.df['conversation_complexity'].mean()
        
        insights.append({
            'category': 'Usage Volume',
            'title': 'Impressive Activity',
            'description': f'Across {total_conversations:,} conversations, {total_messages:,} messages and {total_chars/1_000_000:.1f}M characters were exchanged',
            'impact': 'high'
        })
        
        # Temporal patterns
        df_time = self.df[self.df['create_datetime'].notna()].copy()
        if len(df_time) > 0:
            # Most active hours
            hourly_activity = df_time.groupby('hour_of_day').size()
            peak_hour = hourly_activity.idxmax()
            peak_count = hourly_activity.max()
            
            insights.append({
                'category': 'Temporal Patterns',
                'title': f'Peak activity at {peak_hour}:00',
                'description': f'Maximum activity at {peak_hour}:00 ({peak_count} conversations). This indicates work habits',
                'impact': 'medium'
            })
            
            # Usage evolution
            daily_stats = df_time.groupby(df_time['create_datetime'].dt.date).size()
            if len(daily_stats) > 30:
                recent_activity = daily_stats.tail(30).mean()
                early_activity = daily_stats.head(30).mean()
                growth = (recent_activity - early_activity) / early_activity * 100
                
                insights.append({
                    'category': 'Usage Evolution',
                    'title': f'Activity growth of {growth:.0f}%',
                    'description': f'Comparing first and last 30 days, activity grew by {growth:.0f}%',
                    'impact': 'high'
                })
        
        # Models
        model_stats = self.df['model'].value_counts()
        top_model = model_stats.index[0]
        top_model_percent = (model_stats.iloc[0] / total_conversations) * 100
        
        insights.append({
            'category': 'Model Preferences',
            'title': f'{top_model} - favorite model',
            'description': f'{top_model} is used in {top_model_percent:.0f}% of conversations. Shows preference for quality',
            'impact': 'medium'
        })
        
        # Languages
        lang_stats = self.df['primary_language'].value_counts()
        if len(lang_stats) > 1:
            bilingual_ratio = lang_stats.iloc[1] / lang_stats.iloc[0]
            insights.append({
                'category': 'Multilingual Usage',
            'title': 'Bilingual usage',
            'description': f'Language ratio {lang_stats.index[0]}:{lang_stats.index[1]} = {1/bilingual_ratio:.1f}:1',
                'impact': 'medium'
            })
        
        # Complexity
        complex_dialogs = (self.df['conversation_complexity'] > 7).sum()
        complex_percent = (complex_dialogs / total_conversations) * 100
        
        insights.append({
            'category': 'Task Complexity',
            'title': f'{complex_percent:.0f}% complex conversations',
            'description': f'{complex_dialogs} диалогов имеют высокую сложность (>7). Это продвинутое использование AI',
            'impact': 'high'
        })
        
        # GPT использование
        gpt_usage = self.df['is_gpt'].sum()
        if gpt_usage > 0:
            gpt_percent = (gpt_usage / total_conversations) * 100
            insights.append({
                'category': 'Кастомные GPT',
                'title': f'Экспериментирование с GPTs',
                'description': f'{gpt_usage} диалогов ({gpt_percent:.1f}%) используют кастомные GPT',
                'impact': 'medium'
            })
        
        # Продуктивность
        if 'duration_minutes' in self.df.columns:
            productive_data = self.df[
                (self.df['duration_minutes'] > 0) & 
                (self.df['total_messages'] > 0)
            ].copy()
            
            if len(productive_data) > 0:
                productive_data['efficiency'] = productive_data['total_messages'] / productive_data['duration_minutes']
                avg_efficiency = productive_data['efficiency'].mean()
                
                insights.append({
                    'category': 'Эффективность',
                    'title': f'{avg_efficiency:.1f} сообщений в минуту',
                    'description': 'Высокая скорость взаимодействия говорит об опытном использовании AI',
                    'impact': 'medium'
                })
        
        return insights
    
    def create_dashboard(self):
        """Создает итоговый дашборд"""
        print("🎨 Creating final dashboard...")
        
        # Создаем все визуализации
        time_fig = self.create_time_series_analysis()
        model_fig = self.create_model_analysis()
        patterns_fig = self.create_usage_patterns()
        content_fig = self.create_content_analysis()
        insights = self.generate_insights()
        
        # Сохраняем отдельные графики
        time_fig.write_html("time_analysis.html")
        model_fig.write_html("model_analysis.html")
        patterns_fig.write_html("usage_patterns.html")
        content_fig.write_html("content_analysis.html")
        
        # Создаем сводный HTML отчет
        insights_html = self._create_insights_html(insights)
        
        dashboard_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ChatGPT Analytics Dashboard</title>
            <meta charset="utf-8">
            <style>
                body {{
                    background-color: #1e1e1e;
                    color: #ffffff;
                    font-family: 'Segoe UI', Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                }}
                .header {{
                    text-align: center;
                    padding: 30px 0;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border-radius: 15px;
                    margin-bottom: 30px;
                }}
                .insights-container {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                }}
                .insight-card {{
                    background: rgba(255, 255, 255, 0.1);
                    border-radius: 10px;
                    padding: 20px;
                    border-left: 4px solid #00ff88;
                }}
                .insight-card.high {{
                    border-left-color: #ff6b35;
                }}
                .insight-card.medium {{
                    border-left-color: #4ecdc4;
                }}
                .chart-container {{
                    margin: 30px 0;
                    padding: 20px;
                    background: rgba(255, 255, 255, 0.05);
                    border-radius: 10px;
                }}
                .navigation {{
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    background: rgba(0, 0, 0, 0.8);
                    padding: 15px;
                    border-radius: 10px;
                }}
                .nav-link {{
                    display: block;
                    color: #4ecdc4;
                    text-decoration: none;
                    margin: 5px 0;
                    padding: 5px 10px;
                    border-radius: 5px;
                    transition: background 0.3s;
                }}
                .nav-link:hover {{
                    background: rgba(78, 205, 196, 0.2);
                }}
            </style>
        </head>
        <body>
            <div class="navigation">
                <a href="#insights" class="nav-link">💡 Инсайты</a>
                <a href="time_analysis.html" class="nav-link">📈 Временной анализ</a>
                <a href="model_analysis.html" class="nav-link">🤖 Модели</a>
                <a href="usage_patterns.html" class="nav-link">🔍 Паттерны</a>
                <a href="content_analysis.html" class="nav-link">📝 Контент</a>
            </div>
            
            <div class="header">
                <h1>🚀 ChatGPT Analytics Dashboard</h1>
                <h2>Глубокий анализ {len(self.df):,} диалогов</h2>
                <p>Период: {self.df['create_datetime'].min().strftime('%Y-%m-%d') if 'create_datetime' in self.df.columns else 'N/A'} - {self.df['create_datetime'].max().strftime('%Y-%m-%d') if 'create_datetime' in self.df.columns else 'N/A'}</p>
            </div>
            
            <div id="insights">
                <h2>💡 Ключевые инсайты</h2>
                {insights_html}
            </div>
            
            <div class="chart-container">
                <p>📊 Детальные графики доступны по ссылкам в навигации →</p>
                <p>🎯 Этот дашборд демонстрирует силу аналитики ChatGPT данных</p>
            </div>
            
            <div style="text-align: center; margin-top: 50px; color: #666;">
                <p>Создано с помощью Python, Plotly и любви к данным ❤️</p>
            </div>
        </body>
        </html>
        """
        
        with open('dashboard.html', 'w', encoding='utf-8') as f:
            f.write(dashboard_html)
        
        print("✅ Dashboard created!")
        return insights
    
    def _create_insights_html(self, insights):
        """Создает HTML для инсайтов"""
        html = '<div class="insights-container">'
        
        for insight in insights:
            impact_class = insight.get('impact', 'medium')
            html += f'''
            <div class="insight-card {impact_class}">
                <h3>{insight['title']}</h3>
                <p><strong>{insight['category']}</strong></p>
                <p>{insight['description']}</p>
            </div>
            '''
        
        html += '</div>'
        return html
    
    def run_analysis(self):
        """Запускает полный анализ"""
        print("🚀 Starting complete ChatGPT data analysis...")
        
        self.load_data()
        self.prepare_dataframe()
        insights = self.create_dashboard()
        
        print("\n" + "="*60)
        print("🎉 ANALYSIS COMPLETED!")
        print("="*60)
        print(f"📁 Files created:")
        print("   • dashboard.html - main dashboard")
        print("   • time_analysis.html - time analysis")
        print("   • model_analysis.html - model analysis")
        print("   • usage_patterns.html - usage patterns")
        print("   • content_analysis.html - content analysis")
        print("\n💡 Key insights:")
        
        for insight in insights[:5]:
            print(f"   • {insight['title']}")
        
        print(f"\n🔗 Open dashboard.html for interactive viewing!")
        
        return self


def main():
    analyzer = ChatGPTAnalyzer('conversations.json.zip')
    analyzer.run_analysis()

if __name__ == "__main__":
    main()
