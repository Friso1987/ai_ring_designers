# visualization/dashboard.py

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
import json
from PIL import Image
import base64
from io import BytesIO

class TrainingDashboard:
    """
    Real-time visualization dashboard
    Shows training progress, design evolution, metrics
    """
    
    def __init__(self, trainer=None, port=8050):
        self.trainer = trainer
        self.port = port
        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()
    
    def setup_layout(self):
        """Setup dashboard layout"""
        self.app.layout = html.Div([
            html.H1("🧬 AI Ring Designer - Training Dashboard", 
                   style={'textAlign': 'center', 'color': '#2c3e50'}),
            
            # Auto-refresh
            dcc.Interval(
                id='interval-component',
                interval=5*1000,  # 5 seconds
                n_intervals=0
            ),
            
            # Top row: Key metrics
            html.Div([
                html.Div([
                    html.H3("Current Generation"),
                    html.H2(id='current-generation', children='0'),
                ], className='metric-box'),
                
                html.Div([
                    html.H3("Best Score"),
                    html.H2(id='best-score', children='0.000'),
                ], className='metric-box'),
                
                html.Div([
                    html.H3("Mean Score"),
                    html.H2(id='mean-score', children='0.000'),
                ], className='metric-box'),
                
                html.Div([
                    html.H3("Designs Evaluated"),
                    html.H2(id='total-designs', children='0'),
                ], className='metric-box'),
            ], style={'display': 'flex', 'justifyContent': 'space-around', 'margin': '20px'}),
            
            # Main content area
            html.Div([
                # Left column: Graphs
                html.Div([
                    # Score evolution
                    dcc.Graph(id='score-evolution'),
                    
                    # Loss curves
                    dcc.Graph(id='loss-curves'),
                    
                    # Parameter distribution
                    dcc.Graph(id='parameter-distribution'),
                    
                ], style={'width': '60%', 'display': 'inline-block', 'vertical-align': 'top'}),
                
                # Right column: Best designs gallery
                html.Div([
                    html.H2("🏆 Top Designs"),
                    html.Div(id='best-designs-gallery'),
                    
                    html.Hr(),
                    
                    html.H2("📊 Parameter Analysis"),
                    html.Div(id='parameter-analysis'),
                    
                ], style={'width': '38%', 'display': 'inline-block', 'padding': '20px', 
                         'vertical-align': 'top'}),
            ]),
            
            # Bottom: Design evolution timeline
            html.Div([
                html.H2("🔄 Evolution Timeline"),
                dcc.Graph(id='evolution-timeline'),
            ], style={'margin': '20px'}),
            
        ], style={'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#ecf0f1'})
    
    def setup_callbacks(self):
        """Setup interactive callbacks"""
        
        @self.app.callback(
            [Output('current-generation', 'children'),
             Output('best-score', 'children'),
             Output('mean-score', 'children'),
             Output('total-designs', 'children'),
             Output('score-evolution', 'figure'),
             Output('loss-curves', 'figure'),
             Output('parameter-distribution', 'figure'),
             Output('best-designs-gallery', 'children'),
             Output('parameter-analysis', 'children'),
             Output('evolution-timeline', 'figure')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_dashboard(n):
            if self.trainer is None:
                return self.get_empty_outputs()
            
            # Get current stats
            gen = self.trainer.generation
            best_score = self.trainer.best_score
            buffer_stats = self.trainer.buffer.get_stats()
            mean_score = buffer_stats.get('mean_score', 0)
            total_designs = buffer_stats.get('size', 0)
            
            # Create figures
            score_fig = self.create_score_evolution_plot()
            loss_fig = self.create_loss_curves_plot()
            param_fig = self.create_parameter_distribution_plot()
            gallery = self.create_best_designs_gallery()
            param_analysis = self.create_parameter_analysis()
            timeline_fig = self.create_evolution_timeline()
            
            return (
                str(gen),
                f"{best_score:.3f}",
                f"{mean_score:.3f}",
                str(total_designs),
                score_fig,
                loss_fig,
                param_fig,
                gallery,
                param_analysis,
                timeline_fig
            )
    
    def create_score_evolution_plot(self):
        """Plot score evolution over generations"""
        if not self.trainer.history['generation']:
            return go.Figure()
        
        fig = make_subplots(specs=[[{"secondary_y": False}]])
        
        # Mean score
        fig.add_trace(
            go.Scatter(
                x=self.trainer.history['generation'],
                y=self.trainer.history['mean_score'],
                name='Mean Score',
                line=dict(color='#3498db', width=2),
                mode='lines'
            )
        )
        
        # Max score
        fig.add_trace(
            go.Scatter(
                x=self.trainer.history['generation'],
                y=self.trainer.history['max_score'],
                name='Max Score',
                line=dict(color='#e74c3c', width=2),
                mode='lines'
            )
        )
        
        # Fill between
        fig.add_trace(
            go.Scatter(
                x=self.trainer.history['generation'] + self.trainer.history['generation'][::-1],
                y=self.trainer.history['max_score'] + self.trainer.history['mean_score'][::-1],
                fill='toself',
                fillcolor='rgba(52, 152, 219, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                name='Range'
            )
        )
        
        fig.update_layout(
            title='Score Evolution',
            xaxis_title='Generation',
            yaxis_title='Score',
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def create_loss_curves_plot(self):
        """Plot training losses"""
        if not self.trainer.history['generation']:
            return go.Figure()
        
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Generator Loss', 'Critic Loss'))
        
        # Generator loss
        fig.add_trace(
            go.Scatter(
                x=self.trainer.history['generation'],
                y=self.trainer.history['gen_loss'],
                name='Gen Loss',
                line=dict(color='#9b59b6', width=2)
            ),
            row=1, col=1
        )
        
        # Critic loss
        fig.add_trace(
            go.Scatter(
                x=self.trainer.history['generation'],
                y=self.trainer.history['critic_loss'],
                name='Critic Loss',
                line=dict(color='#e67e22', width=2)
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title='Training Losses',
            hovermode='x unified',
            template='plotly_white',
            showlegend=False
        )
        
        return fig
    
    def create_parameter_distribution_plot(self):
        """Show distribution of parameters in top designs"""
        best_designs = self.trainer.buffer.get_best(n=20)
        
        if not best_designs:
            return go.Figure()
        
        # Extract parameters
        params_list = [d[0] for d in best_designs]
        
        data = {
            'Main Veins': [p.num_main_veins for p in params_list],
            'Lateral Wander': [p.lateral_wander for p in params_list],
            'Branch Prob': [p.branch_probability for p in params_list],
            'Node Density': [p.node_density for p in params_list],
        }
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=list(data.keys())
        )
        
        for i, (name, values) in enumerate(data.items()):
            row = i // 2 + 1
            col = i % 2 + 1
            
            fig.add_trace(
                go.Histogram(x=values, name=name, nbinsx=10),
                row=row, col=col
            )
        
        fig.update_layout(
            title='Parameter Distributions (Top 20 Designs)',
            showlegend=False,
            template='plotly_white'
        )
        
        return fig
    
    def create_best_designs_gallery(self):
        """Create gallery of best designs with images"""
        best_designs = self.trainer.buffer.get_best(n=5)
        
        if not best_designs:
            return html.Div("No designs yet...")
        
        gallery_items = []
        
        for i, (params, score) in enumerate(best_designs):
            # Try to load image if exists
            img_path = Path(f'data/designs/best/best_gen_{self.trainer.generation - i}.png')
            
            if img_path.exists():
                # Convert image to base64
                img = Image.open(img_path)
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                img_data = f"data:image/png;base64,{img_str}"
            else:
                img_data = ""
            
            gallery_items.append(
                html.Div([
                    html.Img(src=img_data, style={'width': '100%', 'borderRadius': '10px'}) if img_data else html.Div(),
                    html.P(f"Score: {score:.3f}", style={'fontWeight': 'bold', 'margin': '5px 0'}),
                    html.P(f"Veins: {params.num_main_veins}, Layers: {params.height_layers}", 
                          style={'fontSize': '12px', 'color': '#7f8c8d'}),
                ], style={
                    'border': '2px solid #3498db' if i == 0 else '1px solid #bdc3c7',
                    'borderRadius': '10px',
                    'padding': '10px',
                    'margin': '10px 0',
                    'backgroundColor': 'white'
                })
            )
        
        return html.Div(gallery_items)
    
    def create_parameter_analysis(self):
        """Analyze which parameters correlate with high scores"""
        buffer_data = [(p, s) for p, s in zip(self.trainer.buffer.buffer, self.trainer.buffer.scores)]
        
        if len(buffer_data) < 10:
            return html.Div("Not enough data yet...")
        
        # Calculate correlations
        params_matrix = np.array([p.to_vector() for p, s in buffer_data])
        scores = np.array([s for p, s in buffer_data])
        
        correlations = []
        param_names = [
            'Main Veins', 'Vein Thick', 'Lateral Wander', 'Radial Wander',
            'Height Layers', 'Twist Factor', 'Branch Bias', 'Branch Prob',
            'Branch Angle', 'Sub Branch Len', 'Node Density', 'Node Size Var',
            'Surface Rough', 'Organic Var'
        ]
        
        for i, name in enumerate(param_names):
            corr = np.corrcoef(params_matrix[:, i], scores)[0, 1]
            correlations.append((name, corr))
        
        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Create visualization
        items = []
        for name, corr in correlations[:8]:  # Top 8
            color = '#27ae60' if corr > 0 else '#e74c3c'
            width = abs(corr) * 100
            
            items.append(
                html.Div([
                    html.Span(name, style={'display': 'inline-block', 'width': '120px'}),
                    html.Div(style={
                        'display': 'inline-block',
                        'width': f'{width}%',
                        'height': '20px',
                        'backgroundColor': color,
                        'borderRadius': '3px'
                    }),
                    html.Span(f" {corr:.3f}", style={'marginLeft': '10px', 'fontSize': '12px'})
                ], style={'margin': '5px 0'})
            )
        
        return html.Div([
            html.P("Parameter correlation with score:", style={'fontWeight': 'bold'}),
            html.Div(items)
        ])
    
    def create_evolution_timeline(self):
        """Show how designs evolved over time"""
        # Get milestone designs (every 10th generation)
        milestones = []
        for gen in range(0, self.trainer.generation + 1, max(1, self.trainer.generation // 10)):
            # Find best design around this generation
            relevant_designs = [
                (p, s, g) for g, (p, s) in enumerate(zip(self.trainer.buffer.buffer, self.trainer.buffer.scores))
                if abs(g - gen) < 5
            ]
            if relevant_designs:
                best = max(relevant_designs, key=lambda x: x[1])
                milestones.append(best)
        
        if not milestones:
            return go.Figure()
        
        # Extract data
        generations = [m[2] for m in milestones]
        scores = [m[1] for m in milestones]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=generations,
            y=scores,
            mode='lines+markers',
            marker=dict(size=12, color=scores, colorscale='Viridis', showscale=True),
            line=dict(width=3, color='#34495e'),
            name='Evolution'
        ))
        
        fig.update_layout(
            title='Design Evolution Timeline',
            xaxis_title='Generation',
            yaxis_title='Score',
            template='plotly_white',
            height=300
        )
        
        return fig
    
    def get_empty_outputs(self):
        """Return empty outputs when no data"""
        return (
            '0', '0.000', '0.000', '0',
            go.Figure(), go.Figure(), go.Figure(),
            html.Div("Waiting for training..."),
            html.Div("Waiting for data..."),
            go.Figure()
        )
    
    def run(self):
        """Start dashboard server"""
        print(f"🚀 Starting dashboard on http://localhost:{self.port}")
        self.app.run(debug=False, port=self.port, use_reloader=False)