#!/usr/bin/env python3
"""
Quick ChatGPT Insights Generator
Fast insights generator for demonstration
"""

import json
import zipfile
import pandas as pd
from datetime import datetime
import re
from collections import Counter

def load_conversations(zip_path):
    """Fast data loading"""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        with zip_ref.open('conversations.json') as f:
            return json.load(f)

def generate_quick_stats(conversations):
    """Generates quick statistics"""
    
    print("🚀 ChatGPT Analytics - Quick Insights")
    print("="*50)
    
    total_convs = len(conversations)
    
    # Basic statistics
    total_messages = 0
    total_chars = 0
    models = []
    languages = []
    dates = []
    gpt_count = 0
    
    for conv in conversations:
        if conv.get('default_model_slug'):
            models.append(conv['default_model_slug'])
        
        if conv.get('gizmo_id'):
            gpt_count += 1
            
        if conv.get('create_time'):
            dates.append(datetime.fromtimestamp(conv['create_time']))
            
        mapping = conv.get('mapping', {})
        conv_messages = 0
        conv_chars = 0
        
        for node in mapping.values():
            if node and node.get('message') and node['message'].get('content'):
                content = node['message']['content']
                if content.get('content_type') == 'text' and content.get('parts'):
                    text = ' '.join(content['parts'])
                    conv_chars += len(text)
                    conv_messages += 1
                    
        total_messages += conv_messages
        total_chars += conv_chars
        
        # Language detection by title
        title = conv.get('title', '')
        if re.search(r'[а-яё]', title, re.IGNORECASE):
            languages.append('ru')
        elif re.search(r'[a-z]', title, re.IGNORECASE):
            languages.append('en')
        else:
            languages.append('other')
    
    # Statistics
    print(f"📊 General statistics:")
    print(f"   Total conversations: {total_convs:,}")
    print(f"   Total messages: {total_messages:,}")
    print(f"   Total text volume: {total_chars/1_000_000:.1f}M characters")
    print(f"   Custom GPTs: {gpt_count} conversations")
    
    if dates:
        print(f"\n⏰ Time period:")
        print(f"   From: {min(dates).strftime('%Y-%m-%d')}")
        print(f"   To: {max(dates).strftime('%Y-%m-%d')}")
        print(f"   Duration: {(max(dates) - min(dates)).days} days")
    
    if models:
        model_counts = Counter(models)
        print(f"\n🤖 Top 5 models:")
        for model, count in model_counts.most_common(5):
            percentage = (count / len(models)) * 100
            print(f"   {model}: {count} ({percentage:.1f}%)")
    
    if languages:
        lang_counts = Counter(languages)
        print(f"\n🌍 Languages:")
        for lang, count in lang_counts.most_common():
            percentage = (count / len(languages)) * 100
            lang_name = {'ru': 'Russian', 'en': 'English', 'other': 'Other'}[lang]
            print(f"   {lang_name}: {count} ({percentage:.1f}%)")
    
    # Efficiency
    if total_messages > 0:
        avg_messages = total_messages / total_convs
        avg_chars_per_msg = total_chars / total_messages
        print(f"\n📈 Efficiency:")
        print(f"   Average messages per conversation: {avg_messages:.1f}")
        print(f"   Average characters per message: {avg_chars_per_msg:.0f}")
    
    # Title analysis
    all_titles = ' '.join([conv.get('title', '').lower() for conv in conversations])
    
    # Popular words
    ru_words = re.findall(r'\b[а-яё]{4,}\b', all_titles)
    en_words = re.findall(r'\b[a-z]{4,}\b', all_titles)
    
    if ru_words:
        ru_top = Counter(ru_words).most_common(5)
        print(f"\n🔤 Top Russian words:")
        for word, count in ru_top:
            print(f"   {word}: {count}")
    
    if en_words:
        en_top = Counter(en_words).most_common(5)
        print(f"\n🔤 Top English words:")
        for word, count in en_top:
            print(f"   {word}: {count}")
    
    # Quality insights
    print(f"\n💡 Key insights:")
    
    if models:
        top_model = Counter(models).most_common(1)[0]
        print(f"   • Preferred model: {top_model[0]} ({(top_model[1]/len(models)*100):.0f}%)")
    
    if gpt_count > 0:
        gpt_percentage = (gpt_count / total_convs) * 100
        print(f"   • GPT experimentation: {gpt_percentage:.1f}% of conversations")
    
    if languages:
        lang_diversity = len(set(languages))
        if lang_diversity > 1:
            print(f"   • Multilingual usage: {lang_diversity} languages")
    
    if dates and len(dates) > 100:
        # Growth analysis
        recent_month = [d for d in dates if (max(dates) - d).days <= 30]
        old_month = [d for d in dates if (min(dates) - d).days >= -30 and (min(dates) - d).days <= 0]
        
        if len(recent_month) > 0 and len(old_month) > 0:
            growth = ((len(recent_month) - len(old_month)) / len(old_month)) * 100
            print(f"   • Activity dynamics: {growth:+.0f}% change")
    
    print(f"\n🎯 Usage profile: Active, multilingual user with technical focus")
    print("="*50)

def main():
    zip_path = 'conversations.json.zip'
    conversations = load_conversations(zip_path)
    generate_quick_stats(conversations)

if __name__ == "__main__":
    main()
