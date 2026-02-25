import re

def parse_and_format_thought(raw_text):
    """
    Transforms a raw stream of AI "thoughts" into a structured, chronological timeline
    mimicking Claude's thinking UI or an agentic orchestration interface.
    """
    if not raw_text:
        return ""
        
    lines = raw_text.split('\n')
    
    # Base container with styling that matches the premium app theme
    html_out = '<div style="font-family: \'Inter\', sans-serif; padding: 4px 0;">'
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
            
        is_header = False
        clean_title = line
        
        # Heuristics to identify a "step" or "header" inside the thought process
        if line.endswith(':'):
            is_header = True
            clean_title = line[:-1].strip().title()
        elif len(line) < 45 and not line.endswith('.') and not line.endswith('?') and not line.startswith('-'):
            is_header = True
            clean_title = line.title()
            
        if is_header:
            # Pick a relevant icon based on keywords
            icon = "⏱️" if i == 0 else "🔹"
            llc = line.lower()
            if "verdict" in llc: icon = "⚖️"
            elif "data" in llc or "metric" in llc: icon = "📊"
            elif "action" in llc or "step" in llc: icon = "🎯"
            elif "risk" in llc: icon = "⚠️"
            elif "thesis" in llc: icon = "💡"
            elif "target" in llc or "stop" in llc: icon = "📌"
            elif "calculat" in llc: icon = "🧮"
            elif "analyz" in llc: icon = "🔍"
            
            html_out += f'''
            <div style="display: flex; align-items: center; margin-top: 14px; margin-bottom: 6px;">
                <span style="margin-right: 10px; font-size: 1.0em; opacity: 0.9;">{icon}</span>
                <span style="font-weight: 600; color: #e2e8f0; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em;">{clean_title}</span>
            </div>
            '''
        else:
            # Regular text or bullet points
            # Ensure it has a vertical timeline line linking to the header
            if line.startswith('-') or line.startswith('*'):
                line_content = line[1:].strip()
                html_out += f'''
                <div style="margin-left: 10px; border-left: 2px solid #334155; padding-left: 18px; margin-bottom: 4px; color: #94a3b8; font-size: 0.85rem; line-height: 1.6;">
                    <span style="color: #64748b; margin-right: 6px;">•</span> {line_content}
                </div>
                '''
            else:
                html_out += f'''
                <div style="margin-left: 10px; border-left: 2px solid #334155; padding-left: 18px; margin-bottom: 6px; color: #94a3b8; font-size: 0.85rem; line-height: 1.6;">
                    {line}
                </div>
                '''
            
    html_out += '</div>'
    return html_out
