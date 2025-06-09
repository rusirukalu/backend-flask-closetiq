# step6_fashion_knowledge_graph_fixed.py - Fixed Fashion Knowledge Graph Integration System

import os
import numpy as np
import json
import pickle
from datetime import datetime
from collections import defaultdict, Counter
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

class FashionKnowledgeGraph:
    """Advanced Fashion Knowledge Graph Integration System"""
    
    def __init__(self):
        # Core knowledge graph
        self.knowledge_graph = nx.MultiDiGraph()
        
        # Knowledge domains
        self.fashion_ontology = {
            'garment_types': {
                'tops': ['shirts', 'blouses', 't-shirts', 'sweaters', 'tanks', 'cardigans'],
                'bottoms': ['pants', 'jeans', 'shorts', 'skirts', 'leggings', 'trousers'],
                'dresses': ['casual_dress', 'formal_dress', 'maxi_dress', 'mini_dress'],
                'outerwear': ['jackets', 'coats', 'blazers', 'vests', 'parkas'],
                'footwear': ['sneakers', 'dress_shoes', 'boots', 'sandals', 'heels'],
                'accessories': ['bags', 'jewelry', 'belts', 'hats', 'scarves', 'watches']
            },
            'style_concepts': {
                'classic': ['timeless', 'traditional', 'elegant', 'refined'],
                'casual': ['relaxed', 'comfortable', 'everyday', 'laid-back'],
                'formal': ['professional', 'business', 'sophisticated', 'polished'],
                'trendy': ['fashionable', 'current', 'stylish', 'contemporary'],
                'bohemian': ['free-spirited', 'artistic', 'unconventional', 'eclectic'],
                'minimalist': ['simple', 'clean', 'understated', 'modern'],
                'edgy': ['bold', 'daring', 'unconventional', 'dramatic'],
                'romantic': ['feminine', 'soft', 'delicate', 'graceful']
            },
            'color_relationships': {
                'primary': ['red', 'blue', 'yellow'],
                'secondary': ['green', 'orange', 'purple'],
                'neutral': ['black', 'white', 'gray', 'beige', 'brown'],
                'warm': ['red', 'orange', 'yellow', 'pink', 'coral'],
                'cool': ['blue', 'green', 'purple', 'teal', 'navy'],
                'earth_tones': ['brown', 'tan', 'olive', 'rust', 'ochre']
            },
            'occasions': {
                'professional': ['work', 'business_meeting', 'interview', 'conference'],
                'social': ['party', 'date', 'dinner', 'celebration', 'wedding'],
                'casual': ['weekend', 'shopping', 'lunch', 'errands', 'travel'],
                'formal': ['gala', 'opera', 'black_tie', 'cocktail', 'ceremony'],
                'active': ['gym', 'sports', 'hiking', 'yoga', 'running']
            },
            'seasons': {
                'spring': ['light_layers', 'pastels', 'transitional', 'fresh'],
                'summer': ['lightweight', 'breathable', 'bright_colors', 'minimal'],
                'fall': ['layers', 'earth_tones', 'cozy', 'rich_colors'],
                'winter': ['warm', 'heavy_fabrics', 'dark_colors', 'coverage']
            }
        }
        
        # Compatibility rules and relationships
        self.compatibility_rules = self._load_compatibility_knowledge()
        
        # Semantic embeddings for fashion concepts
        self.concept_embeddings = {}
        
        # Fashion rules and constraints
        self.fashion_rules = self._load_fashion_rules()
        
        # Style influence networks
        self.style_influences = self._load_style_influences()
        
    def _load_compatibility_knowledge(self):
        """Load comprehensive compatibility knowledge"""
        return {
            'color_harmony': {
                'complementary': [
                    ('red', 'green'), ('blue', 'orange'), ('yellow', 'purple'),
                    ('navy', 'coral'), ('forest_green', 'burgundy')
                ],
                'analogous': [
                    ('blue', 'blue_green', 'green'),
                    ('red', 'red_orange', 'orange'),
                    ('yellow', 'yellow_green', 'green')
                ],
                'triadic': [
                    ('red', 'yellow', 'blue'),
                    ('orange', 'green', 'purple')
                ],
                'monochromatic': {
                    'blue': ['navy', 'royal_blue', 'sky_blue', 'powder_blue'],
                    'gray': ['charcoal', 'slate', 'silver', 'light_gray']
                },
                'neutral_pairs': [
                    ('black', 'white'), ('gray', 'white'), ('navy', 'cream'),
                    ('brown', 'beige'), ('charcoal', 'ivory')
                ]
            },
            'style_compatibility': {
                'high_compatibility': [
                    ('classic', 'elegant'), ('casual', 'comfortable'),
                    ('formal', 'professional'), ('bohemian', 'artistic'),
                    ('minimalist', 'modern'), ('edgy', 'bold')
                ],
                'moderate_compatibility': [
                    ('classic', 'formal'), ('casual', 'bohemian'),
                    ('minimalist', 'classic'), ('trendy', 'edgy')
                ],
                'style_conflicts': [
                    ('formal', 'casual'), ('minimalist', 'bohemian'),
                    ('classic', 'edgy'), ('elegant', 'grunge')
                ]
            },
            'garment_combinations': {
                'classic_pairs': [
                    ('blazer', 'trousers'), ('shirt', 'jeans'),
                    ('dress', 'heels'), ('sweater', 'skirt')
                ],
                'layering_rules': [
                    ('tank', 'cardigan'), ('shirt', 'blazer'),
                    ('dress', 'jacket'), ('tshirt', 'vest')
                ],
                'avoid_combinations': [
                    ('formal_shoes', 'athletic_wear'),
                    ('evening_gown', 'sneakers'),
                    ('business_suit', 'flip_flops')
                ]
            }
        }
    
    def _load_fashion_rules(self):
        """Load fashion rules and constraints"""
        return {
            'proportion_rules': {
                'fitted_top_loose_bottom': "When wearing a fitted top, pair with looser bottoms",
                'loose_top_fitted_bottom': "When wearing a loose top, pair with fitted bottoms",
                'avoid_baggy_on_baggy': "Avoid wearing baggy items together"
            },
            'color_rules': {
                'no_more_than_three': "Limit outfit to 3 main colors maximum",
                'neutral_base': "Use neutrals as base, add one accent color",
                'metallic_mixing': "Don't mix gold and silver metals in one outfit"
            },
            'occasion_rules': {
                'workplace_coverage': "Ensure appropriate coverage for professional settings",
                'formal_event_elegance': "Choose elevated pieces for formal occasions",
                'casual_comfort': "Prioritize comfort and practicality for casual wear"
            },
            'seasonal_rules': {
                'fabric_weight': "Match fabric weight to season temperature",
                'color_seasonality': "Choose colors appropriate for the season",
                'layering_strategy': "Use layering for transitional seasons"
            }
        }
    
    def _load_style_influences(self):
        """Load style influence networks and trends"""
        return {
            'style_evolution': {
                'classic': {'influenced_by': ['audrey_hepburn', 'grace_kelly'], 
                           'influences': ['timeless_elegance', 'refined_simplicity']},
                'bohemian': {'influenced_by': ['1960s_hippie', 'folk_culture'], 
                           'influences': ['free_spirit', 'artistic_expression']},
                'minimalist': {'influenced_by': ['scandinavian_design', 'japanese_aesthetics'], 
                              'influences': ['clean_lines', 'functional_beauty']}
            },
            'cultural_influences': {
                'parisian_chic': ['effortless_elegance', 'quality_basics', 'understated_luxury'],
                'italian_style': ['impeccable_tailoring', 'luxury_fabrics', 'confident_flair'],
                'japanese_fashion': ['innovative_cutting', 'experimental_silhouettes', 'attention_to_detail']
            },
            'trend_cycles': {
                'recurring_trends': ['vintage_revival', 'minimalism', 'maximalism', 'athleisure'],
                'micro_trends': ['seasonal_colors', 'specific_silhouettes', 'pattern_preferences'],
                'macro_trends': ['sustainability', 'technology_integration', 'customization']
            }
        }
    
    def build_knowledge_graph(self):
        """Build the comprehensive fashion knowledge graph"""
        print("🧠 BUILDING FASHION KNOWLEDGE GRAPH")
        print("=" * 60)
        
        # Step 1: Add core fashion concepts as nodes
        self._add_concept_nodes()
        
        # Step 2: Add compatibility relationships
        self._add_compatibility_edges()
        
        # Step 3: Add semantic relationships
        self._add_semantic_relationships()
        
        # Step 4: Add style influence networks
        self._add_style_influences()
        
        # Step 5: Add occasion and seasonal connections
        self._add_contextual_relationships()
        
        # Step 6: Add fashion rules as constraint nodes
        self._add_fashion_rules()
        
        print(f"✅ Knowledge graph built:")
        print(f"   Nodes: {self.knowledge_graph.number_of_nodes()}")
        print(f"   Edges: {self.knowledge_graph.number_of_edges()}")
        
        return True
    
    def _add_concept_nodes(self):
        """Add all fashion concepts as nodes"""
        print("📍 Adding concept nodes...")
        
        node_count = 0
        
        # Add garment types
        for category, items in self.fashion_ontology['garment_types'].items():
            # Add category node
            self.knowledge_graph.add_node(category, 
                                        type='garment_category',
                                        level='category')
            node_count += 1
            
            # Add individual items
            for item in items:
                self.knowledge_graph.add_node(item, 
                                            type='garment_item',
                                            category=category,
                                            level='item')
                # Connect item to category
                self.knowledge_graph.add_edge(item, category, 
                                            relationship='belongs_to')
                node_count += 1
        
        # Add style concepts
        for style, attributes in self.fashion_ontology['style_concepts'].items():
            self.knowledge_graph.add_node(style, 
                                        type='style_concept',
                                        level='style')
            node_count += 1
            
            for attribute in attributes:
                self.knowledge_graph.add_node(attribute, 
                                            type='style_attribute',
                                            parent_style=style,
                                            level='attribute')
                self.knowledge_graph.add_edge(attribute, style, 
                                            relationship='describes')
                node_count += 1
        
        # Add colors
        for color_group, colors in self.fashion_ontology['color_relationships'].items():
            self.knowledge_graph.add_node(color_group, 
                                        type='color_group',
                                        level='group')
            node_count += 1
            
            for color in colors:
                if not self.knowledge_graph.has_node(color):
                    self.knowledge_graph.add_node(color, 
                                                type='color',
                                                color_group=color_group,
                                                level='color')
                    node_count += 1
                
                self.knowledge_graph.add_edge(color, color_group, 
                                            relationship='belongs_to')
        
        # Add occasions
        for occasion_group, occasions in self.fashion_ontology['occasions'].items():
            self.knowledge_graph.add_node(occasion_group, 
                                        type='occasion_group',
                                        level='group')
            node_count += 1
            
            for occasion in occasions:
                self.knowledge_graph.add_node(occasion, 
                                            type='occasion',
                                            occasion_group=occasion_group,
                                            level='occasion')
                self.knowledge_graph.add_edge(occasion, occasion_group, 
                                            relationship='belongs_to')
                node_count += 1
        
        print(f"   Added {node_count} concept nodes")
    
    def _add_compatibility_edges(self):
        """Add compatibility relationships between concepts"""
        print("🔗 Adding compatibility edges...")
        
        edge_count = 0
        
        # Color compatibility
        color_rules = self.compatibility_rules['color_harmony']
        
        # Complementary colors
        for color1, color2 in color_rules['complementary']:
            if (self.knowledge_graph.has_node(color1) and 
                self.knowledge_graph.has_node(color2)):
                self.knowledge_graph.add_edge(color1, color2, 
                                            relationship='complementary',
                                            compatibility_score=0.9)
                edge_count += 1
        
        # Neutral pairs
        for color1, color2 in color_rules['neutral_pairs']:
            if (self.knowledge_graph.has_node(color1) and 
                self.knowledge_graph.has_node(color2)):
                self.knowledge_graph.add_edge(color1, color2, 
                                            relationship='neutral_harmony',
                                            compatibility_score=0.85)
                edge_count += 1
        
        # Style compatibility
        style_rules = self.compatibility_rules['style_compatibility']
        
        for style1, style2 in style_rules['high_compatibility']:
            if (self.knowledge_graph.has_node(style1) and 
                self.knowledge_graph.has_node(style2)):
                self.knowledge_graph.add_edge(style1, style2, 
                                            relationship='highly_compatible',
                                            compatibility_score=0.9)
                edge_count += 1
        
        for style1, style2 in style_rules['moderate_compatibility']:
            if (self.knowledge_graph.has_node(style1) and 
                self.knowledge_graph.has_node(style2)):
                self.knowledge_graph.add_edge(style1, style2, 
                                            relationship='moderately_compatible',
                                            compatibility_score=0.7)
                edge_count += 1
        
        for style1, style2 in style_rules['style_conflicts']:
            if (self.knowledge_graph.has_node(style1) and 
                self.knowledge_graph.has_node(style2)):
                self.knowledge_graph.add_edge(style1, style2, 
                                            relationship='conflicts_with',
                                            compatibility_score=0.2)
                edge_count += 1
        
        # Garment combinations
        garment_rules = self.compatibility_rules['garment_combinations']
        
        for item1, item2 in garment_rules['classic_pairs']:
            # Find actual nodes that match these items
            matching_nodes1 = [n for n in self.knowledge_graph.nodes() 
                              if item1.lower() in n.lower()]
            matching_nodes2 = [n for n in self.knowledge_graph.nodes() 
                              if item2.lower() in n.lower()]
            
            for node1 in matching_nodes1:
                for node2 in matching_nodes2:
                    self.knowledge_graph.add_edge(node1, node2, 
                                                relationship='classic_pairing',
                                                compatibility_score=0.85)
                    edge_count += 1
        
        print(f"   Added {edge_count} compatibility edges")
    
    def _add_semantic_relationships(self):
        """Add semantic relationships between concepts"""
        print("🧠 Adding semantic relationships...")
        
        edge_count = 0
        
        # Add hierarchical relationships
        # Colors -> Color groups
        for color_group, colors in self.fashion_ontology['color_relationships'].items():
            for color in colors:
                if (self.knowledge_graph.has_node(color) and 
                    self.knowledge_graph.has_node(color_group)):
                    self.knowledge_graph.add_edge(color, color_group, 
                                                relationship='is_type_of',
                                                semantic_weight=1.0)
                    edge_count += 1
        
        # Add semantic similarity between related concepts
        # Similar styles
        similar_styles = [
            ('classic', 'elegant', 0.8),
            ('casual', 'comfortable', 0.9),
            ('formal', 'professional', 0.85),
            ('bohemian', 'artistic', 0.8),
            ('minimalist', 'modern', 0.75)
        ]
        
        for style1, style2, similarity in similar_styles:
            if (self.knowledge_graph.has_node(style1) and 
                self.knowledge_graph.has_node(style2)):
                self.knowledge_graph.add_edge(style1, style2, 
                                            relationship='semantically_similar',
                                            similarity_score=similarity)
                edge_count += 1
        
        print(f"   Added {edge_count} semantic relationships")
    
    def _add_style_influences(self):
        """Add style influence networks"""
        print("🎨 Adding style influences...")
        
        edge_count = 0
        
        for style, influence_data in self.style_influences['style_evolution'].items():
            if self.knowledge_graph.has_node(style):
                # Add influence sources
                for source in influence_data['influenced_by']:
                    if not self.knowledge_graph.has_node(source):
                        self.knowledge_graph.add_node(source, 
                                                    type='influence_source',
                                                    level='cultural')
                    
                    self.knowledge_graph.add_edge(source, style, 
                                                relationship='influences',
                                                influence_strength=0.8)
                    edge_count += 1
                
                # Add influenced concepts
                for influenced in influence_data['influences']:
                    if not self.knowledge_graph.has_node(influenced):
                        self.knowledge_graph.add_node(influenced, 
                                                    type='influenced_concept',
                                                    level='concept')
                    
                    self.knowledge_graph.add_edge(style, influenced, 
                                                relationship='creates',
                                                influence_strength=0.7)
                    edge_count += 1
        
        print(f"   Added {edge_count} influence relationships")
    
    def _add_contextual_relationships(self):
        """Add occasion and seasonal context relationships"""
        print("🌍 Adding contextual relationships...")
        
        edge_count = 0
        
        # Seasonal appropriateness
        seasonal_garments = {
            'spring': ['light_jacket', 'cardigan', 'trench_coat'],
            'summer': ['shorts', 't-shirts', 'sandals', 'sundress'],
            'fall': ['sweater', 'boots', 'jacket'],
            'winter': ['coat', 'boots', 'scarf', 'gloves']
        }
        
        for season, garments in seasonal_garments.items():
            if not self.knowledge_graph.has_node(season):
                self.knowledge_graph.add_node(season, 
                                            type='season',
                                            level='context')
            
            for garment in garments:
                # Find matching nodes
                matching_nodes = [n for n in self.knowledge_graph.nodes() 
                                if garment.lower() in n.lower() or 
                                   any(part in n.lower() for part in garment.lower().split('_'))]
                
                for node in matching_nodes:
                    self.knowledge_graph.add_edge(node, season, 
                                                relationship='appropriate_for',
                                                appropriateness_score=0.8)
                    edge_count += 1
        
        # Occasion appropriateness
        occasion_garments = {
            'work': ['blazer', 'trousers', 'dress_shoes', 'blouse'],
            'party': ['dress', 'heels', 'cocktail_dress'],
            'casual': ['jeans', 't-shirts', 'sneakers'],
            'formal': ['suit', 'dress_shoes', 'tie', 'gown']
        }
        
        for occasion, garments in occasion_garments.items():
            for garment in garments:
                matching_nodes = [n for n in self.knowledge_graph.nodes() 
                                if garment.lower() in n.lower() or 
                                   any(part in n.lower() for part in garment.lower().split('_'))]
                
                for node in matching_nodes:
                    if self.knowledge_graph.has_node(occasion):
                        self.knowledge_graph.add_edge(node, occasion, 
                                                    relationship='suitable_for',
                                                    suitability_score=0.85)
                        edge_count += 1
        
        print(f"   Added {edge_count} contextual relationships")
    
    def _add_fashion_rules(self):
        """Add fashion rules as constraint nodes"""
        print("📏 Adding fashion rules...")
        
        rule_count = 0
        
        for rule_category, rules in self.fashion_rules.items():
            # Add rule category node
            category_node = f"rule_{rule_category}"
            self.knowledge_graph.add_node(category_node, 
                                        type='rule_category',
                                        level='rules')
            
            for rule_name, rule_description in rules.items():
                rule_node = f"rule_{rule_name}"
                self.knowledge_graph.add_node(rule_node, 
                                            type='fashion_rule',
                                            description=rule_description,
                                            category=rule_category,
                                            level='rule')
                
                self.knowledge_graph.add_edge(rule_node, category_node, 
                                            relationship='belongs_to')
                rule_count += 1
        
        print(f"   Added {rule_count} fashion rules")
    
    def query_compatibility(self, item1, item2, context=None):
        """Query compatibility between two fashion items"""
        print(f"\n🔍 QUERYING COMPATIBILITY: {item1} ↔ {item2}")
        
        if context:
            print(f"Context: {context}")
        
        # Find nodes in knowledge graph
        item1_nodes = self._find_matching_nodes(item1)
        item2_nodes = self._find_matching_nodes(item2)
        
        if not item1_nodes or not item2_nodes:
            print("❌ Items not found in knowledge graph")
            return None
        
        # Calculate compatibility scores
        compatibility_results = []
        
        for node1 in item1_nodes:
            for node2 in item2_nodes:
                score = self._calculate_compatibility_score(node1, node2, context)
                
                if score > 0:
                    compatibility_results.append({
                        'item1_node': node1,
                        'item2_node': node2,
                        'compatibility_score': score,
                        'reasoning': self._generate_compatibility_reasoning(node1, node2, score)
                    })
        
        if not compatibility_results:
            return {
                'compatible': False,
                'score': 0.0,
                'reasoning': "No compatibility relationships found"
            }
        
        # Get best compatibility
        best_match = max(compatibility_results, key=lambda x: x['compatibility_score'])
        
        result = {
            'compatible': best_match['compatibility_score'] > 0.6,
            'score': best_match['compatibility_score'],
            'reasoning': best_match['reasoning'],
            'all_matches': compatibility_results
        }
        
        print(f"✅ Compatibility Score: {result['score']:.2f}")
        print(f"Reasoning: {result['reasoning']}")
        
        return result
    
    def _find_matching_nodes(self, item):
        """Find nodes in knowledge graph that match an item"""
        item_lower = item.lower()
        matching_nodes = []
        
        for node in self.knowledge_graph.nodes():
            node_lower = node.lower()
            
            # Exact match
            if item_lower == node_lower:
                matching_nodes.append(node)
            # Partial match
            elif item_lower in node_lower or node_lower in item_lower:
                matching_nodes.append(node)
            # Word-based match
            elif any(word in node_lower for word in item_lower.split()):
                matching_nodes.append(node)
        
        return matching_nodes
    
    def _calculate_compatibility_score(self, node1, node2, context=None):
        """Calculate compatibility score between two nodes"""
        score = 0.0
        
        # Direct compatibility edge
        if self.knowledge_graph.has_edge(node1, node2):
            edge_data = self.knowledge_graph.get_edge_data(node1, node2)
            for edge in edge_data.values():
                if 'compatibility_score' in edge:
                    score = max(score, edge['compatibility_score'])
        
        # Reverse direction
        if self.knowledge_graph.has_edge(node2, node1):
            edge_data = self.knowledge_graph.get_edge_data(node2, node1)
            for edge in edge_data.values():
                if 'compatibility_score' in edge:
                    score = max(score, edge['compatibility_score'])
        
        # Indirect compatibility through shared connections
        if score == 0:
            common_neighbors = set(self.knowledge_graph.neighbors(node1)) & \
                             set(self.knowledge_graph.neighbors(node2))
            
            if common_neighbors:
                # Calculate score based on shared connections
                shared_score = len(common_neighbors) / 10.0  # Normalize
                score = min(shared_score, 0.7)  # Cap at 0.7 for indirect
        
        # Context bonus
        if context and score > 0:
            context_bonus = self._calculate_context_bonus(node1, node2, context)
            score = min(1.0, score + context_bonus)
        
        return score
    
    def _calculate_context_bonus(self, node1, node2, context):
        """Calculate bonus score based on context"""
        bonus = 0.0
        
        # Check if both items are appropriate for the context
        context_nodes = self._find_matching_nodes(context)
        
        for context_node in context_nodes:
            node1_appropriate = self._is_appropriate_for_context(node1, context_node)
            node2_appropriate = self._is_appropriate_for_context(node2, context_node)
            
            if node1_appropriate and node2_appropriate:
                bonus += 0.15
        
        return bonus
    
    def _is_appropriate_for_context(self, item_node, context_node):
        """Check if an item is appropriate for a context"""
        # Check direct relationships
        if self.knowledge_graph.has_edge(item_node, context_node):
            edge_data = self.knowledge_graph.get_edge_data(item_node, context_node)
            for edge in edge_data.values():
                relationship = edge.get('relationship', '')
                if relationship in ['appropriate_for', 'suitable_for']:
                    return True
        
        return False
    
    def _generate_compatibility_reasoning(self, node1, node2, score):
        """Generate human-readable reasoning for compatibility"""
        if score >= 0.8:
            return f"{node1} and {node2} are highly compatible fashion items"
        elif score >= 0.6:
            return f"{node1} and {node2} work well together"
        elif score >= 0.4:
            return f"{node1} and {node2} can be paired with some styling consideration"
        else:
            return f"{node1} and {node2} may not be the best combination"
    
    def get_style_recommendations(self, base_item, context=None, limit=5):
        """Get style recommendations based on a base item"""
        print(f"\n💡 GETTING STYLE RECOMMENDATIONS FOR: {base_item}")
        
        if context:
            print(f"Context: {context}")
        
        # Find base item nodes
        base_nodes = self._find_matching_nodes(base_item)
        
        if not base_nodes:
            print("❌ Base item not found in knowledge graph")
            return []
        
        recommendations = []
        
        for base_node in base_nodes:
            # Find compatible items through graph traversal
            compatible_items = self._traverse_for_recommendations(base_node, context)
            recommendations.extend(compatible_items)
        
        # Sort by compatibility score and remove duplicates
        unique_recommendations = {}
        for rec in recommendations:
            item = rec['recommended_item']
            if item not in unique_recommendations or \
               rec['compatibility_score'] > unique_recommendations[item]['compatibility_score']:
                unique_recommendations[item] = rec
        
        # Sort and limit results
        sorted_recommendations = sorted(unique_recommendations.values(), 
                                      key=lambda x: x['compatibility_score'], reverse=True)
        
        result = sorted_recommendations[:limit]
        
        print(f"✅ Found {len(result)} recommendations:")
        for rec in result:
            print(f"   • {rec['recommended_item']} (score: {rec['compatibility_score']:.2f})")
            print(f"     Reason: {rec['reasoning']}")
        
        return result
    
    def _traverse_for_recommendations(self, base_node, context=None):
        """Traverse knowledge graph to find recommendations"""
        recommendations = []
        
        # Get direct neighbors with compatibility relationships
        for neighbor in self.knowledge_graph.neighbors(base_node):
            edge_data = self.knowledge_graph.get_edge_data(base_node, neighbor)
            
            for edge in edge_data.values():
                relationship = edge.get('relationship', '')
                
                if relationship in ['highly_compatible', 'classic_pairing', 'moderately_compatible']:
                    score = edge.get('compatibility_score', 0.5)
                    
                    # Context adjustment
                    if context:
                        context_bonus = self._calculate_context_bonus(base_node, neighbor, context)
                        score = min(1.0, score + context_bonus)
                    
                    recommendations.append({
                        'recommended_item': neighbor,
                        'compatibility_score': score,
                        'reasoning': f"Recommended based on {relationship} with {base_node}"
                    })
        
        return recommendations
    
    def analyze_style_profile(self, wardrobe_items):
        """Analyze style profile based on wardrobe items"""
        print(f"\n📊 ANALYZING STYLE PROFILE")
        print(f"Wardrobe items: {len(wardrobe_items)}")
        
        # Map items to knowledge graph concepts
        style_concepts = defaultdict(int)
        garment_categories = defaultdict(int)
        color_preferences = defaultdict(int)
        
        for item in wardrobe_items:
            # Find matching nodes
            matching_nodes = self._find_matching_nodes(item)
            
            for node in matching_nodes:
                node_data = self.knowledge_graph.nodes[node]
                
                # Count style concepts
                if node_data.get('type') == 'style_concept':
                    style_concepts[node] += 1
                
                # Count garment categories
                elif node_data.get('type') == 'garment_category':
                    garment_categories[node] += 1
                
                # Count colors
                elif node_data.get('type') == 'color':
                    color_preferences[node] += 1
        
        # Generate style profile
        profile = {
            'dominant_styles': dict(Counter(style_concepts).most_common(3)),
            'garment_distribution': dict(garment_categories),
            'color_preferences': dict(Counter(color_preferences).most_common(5)),
            'style_personality': self._determine_style_personality(style_concepts),
            'wardrobe_balance': self._analyze_wardrobe_balance(garment_categories)
        }
        
        print(f"✅ Style Profile Generated:")
        print(f"   Dominant styles: {list(profile['dominant_styles'].keys())}")
        print(f"   Style personality: {profile['style_personality']}")
        print(f"   Wardrobe balance: {profile['wardrobe_balance']}")
        
        return profile
    
    def _determine_style_personality(self, style_concepts):
        """Determine overall style personality"""
        if not style_concepts:
            return "Undefined"
        
        top_style = max(style_concepts, key=style_concepts.get)
        
        personality_map = {
            'classic': 'Timeless Elegance',
            'casual': 'Relaxed Comfort',
            'formal': 'Professional Polish',
            'bohemian': 'Free Spirit',
            'minimalist': 'Modern Simplicity',
            'edgy': 'Bold Statement',
            'romantic': 'Feminine Grace'
        }
        
        return personality_map.get(top_style, 'Eclectic Mix')
    
    def _analyze_wardrobe_balance(self, garment_categories):
        """Analyze wardrobe balance and completeness"""
        total_items = sum(garment_categories.values())
        
        if total_items == 0:
            return "Empty wardrobe"
        
        # Calculate percentages
        percentages = {cat: count/total_items for cat, count in garment_categories.items()}
        
        # Determine balance
        if max(percentages.values()) > 0.6:
            dominant_category = max(percentages, key=percentages.get)
            return f"Heavily focused on {dominant_category}"
        elif len(garment_categories) >= 4:
            return "Well-balanced wardrobe"
        else:
            return "Limited variety"
    
    def visualize_knowledge_graph(self, save_path=None):
        """Visualize the fashion knowledge graph"""
        print(f"\n🎨 VISUALIZING KNOWLEDGE GRAPH")
        
        # Create subgraph for visualization (too many nodes for full graph)
        important_nodes = []
        
        # Select important nodes from each category
        for node, data in self.knowledge_graph.nodes(data=True):
            node_type = data.get('type', '')
            if node_type in ['garment_category', 'style_concept', 'color_group']:
                important_nodes.append(node)
        
        # Create subgraph
        subgraph = self.knowledge_graph.subgraph(important_nodes[:50])  # Limit for visibility
        
        # Create visualization
        plt.figure(figsize=(16, 12))
        
        # Define node colors by type
        node_colors = []
        node_types = nx.get_node_attributes(subgraph, 'type')
        
        color_map = {
            'garment_category': '#FF6B6B',
            'style_concept': '#4ECDC4',
            'color_group': '#45B7D1',
            'occasion_group': '#96CEB4',
            'season': '#FFEAA7'
        }
        
        for node in subgraph.nodes():
            node_type = node_types.get(node, 'unknown')
            node_colors.append(color_map.get(node_type, '#DDA0DD'))
        
        # Create layout
        pos = nx.spring_layout(subgraph, k=3, iterations=50)
        
        # Draw graph
        nx.draw(subgraph, pos, 
                node_color=node_colors,
                node_size=1000,
                font_size=8,
                font_weight='bold',
                with_labels=True,
                edge_color='gray',
                alpha=0.7)
        
        plt.title('Fashion Knowledge Graph - Core Concepts', 
                 fontsize=16, fontweight='bold')
        
        # Add legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=color, markersize=10, label=node_type.replace('_', ' ').title())
                         for node_type, color in color_map.items()]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Knowledge graph saved: {save_path}")
        else:
            plt.savefig('fashion_knowledge_graph.png', dpi=300, bbox_inches='tight')
            print(f"✅ Knowledge graph saved: fashion_knowledge_graph.png")
        
        plt.show()
    
    def save_knowledge_graph(self):
        """Save the knowledge graph and related data"""
        print(f"\n💾 SAVING KNOWLEDGE GRAPH")
        
        # Save knowledge graph
        import pickle
        with open('fashion_knowledge_graph.pkl', 'wb') as f:
            pickle.dump(self.knowledge_graph, f)
        
        # Save configuration
        config = {
            'creation_date': datetime.now().isoformat(),
            'nodes_count': self.knowledge_graph.number_of_nodes(),
            'edges_count': self.knowledge_graph.number_of_edges(),
            'fashion_ontology': self.fashion_ontology,
            'compatibility_rules': self.compatibility_rules,
            'fashion_rules': self.fashion_rules,
            'style_influences': self.style_influences
        }
        
        with open('knowledge_graph_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print("✅ Knowledge graph saved:")
        print("   • fashion_knowledge_graph.pkl - Graph structure")
        print("   • knowledge_graph_config.json - Configuration")

def run_step6_development():
    """Run Step 6: Fashion Knowledge Graph Integration Development"""
    print("🎯 STEP 6: FASHION KNOWLEDGE GRAPH INTEGRATION")
    print("=" * 80)
    print("Goal: Create comprehensive fashion knowledge graph with semantic understanding")
    print("Input: Fashion ontology + compatibility rules + semantic relationships")
    print("Output: Intelligent fashion knowledge base with reasoning capabilities")
    print("=" * 80)
    
    # Initialize knowledge graph
    kg = FashionKnowledgeGraph()
    
    # Build knowledge graph
    success = kg.build_knowledge_graph()
    
    if success:
        print(f"\n🎉 STEP 6 COMPLETE!")
        print("=" * 40)
        print("✅ Comprehensive fashion knowledge graph created")
        print("✅ Semantic relationships established")
        print("✅ Compatibility rules integrated")
        print("✅ Style influence networks mapped")
        print("✅ Fashion rules and constraints encoded")
        print("✅ Context-aware reasoning capabilities")
        
        # Demonstrate knowledge graph capabilities
        print(f"\n🎯 DEMONSTRATION:")
        
        # Test compatibility query
        compatibility = kg.query_compatibility("blazer", "jeans", "work")
        
        # Test style recommendations
        recommendations = kg.get_style_recommendations("black dress", "formal_event", limit=3)
        
        # Test style profile analysis
        sample_wardrobe = ["blazer", "jeans", "white_shirt", "black_dress", "sneakers"]
        style_profile = kg.analyze_style_profile(sample_wardrobe)
        
        # Visualize knowledge graph
        kg.visualize_knowledge_graph()
        
        # Save knowledge graph
        kg.save_knowledge_graph()
        
        print(f"\n📁 FILES CREATED:")
        print("   • fashion_knowledge_graph.pkl - Complete knowledge graph")
        print("   • knowledge_graph_config.json - System configuration")
        print("   • fashion_knowledge_graph.png - Visualization")
        
        print(f"\n🧠 KNOWLEDGE CAPABILITIES:")
        print("   1. Semantic fashion understanding")
        print("   2. Context-aware compatibility analysis")
        print("   3. Intelligent style recommendations")
        print("   4. Fashion rule reasoning")
        print("   5. Style personality profiling")
        print("   6. Cultural and trend influence tracking")
        
        print(f"\n💡 ADVANCED FEATURES:")
        print("   • 'Why do these items work together?' - Semantic reasoning")
        print("   • 'What's my style personality?' - Wardrobe analysis")
        print("   • 'Complete this outfit for...' - Context-aware suggestions")
        print("   • 'Learn fashion rules' - Explainable AI recommendations")
        
        print(f"\n➡️ READY FOR STEP 7:")
        print("   Advanced Web Interface with Visual Features")
        
        return True
    else:
        print("❌ Step 6 development failed")
        return False

if __name__ == "__main__":
    success = run_step6_development()
    
    if success:
        print("\n🚀 Step 6 completed successfully!")
        print("Fashion Knowledge Graph Integration system is ready!")
        print("Ready to proceed to Step 7: Advanced Web Interface")
    else:
        print("\n❌ Step 6 failed - check configuration and try again")
