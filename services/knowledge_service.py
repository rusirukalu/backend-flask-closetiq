# services/knowledge_service.py - Fashion Knowledge Graph Service
import networkx as nx
import pickle
import json
import os
import numpy as np
from datetime import datetime
import logging
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)

class FashionKnowledgeService:
    """Fashion Knowledge Graph for Semantic Intelligence"""
    
    def __init__(self, knowledge_graph_path):
        self.knowledge_graph_path = knowledge_graph_path
        self.knowledge_graph = nx.MultiDiGraph()
        
        # Fashion ontology
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
        
        # Load or build knowledge graph
        self.load_or_build_knowledge_graph()
        
    def load_or_build_knowledge_graph(self):
        """Load existing knowledge graph or build new one"""
        try:
            if os.path.exists(self.knowledge_graph_path):
                with open(self.knowledge_graph_path, 'rb') as f:
                    self.knowledge_graph = pickle.load(f)
                logger.info(f"✅ Knowledge graph loaded: {self.knowledge_graph.number_of_nodes()} nodes, {self.knowledge_graph.number_of_edges()} edges")
            else:
                logger.info("Building new knowledge graph...")
                self.build_knowledge_graph()
                self.save_knowledge_graph()
        except Exception as e:
            logger.error(f"❌ Error loading knowledge graph: {str(e)}")
            # Build new graph as fallback
            self.build_knowledge_graph()
    
    def build_knowledge_graph(self):
        """Build comprehensive fashion knowledge graph"""
        try:
            logger.info("🧠 Building fashion knowledge graph...")
            
            # Add concept nodes
            self._add_concept_nodes()
            
            # Add compatibility relationships
            self._add_compatibility_edges()
            
            # Add semantic relationships
            self._add_semantic_relationships()
            
            # Add style influences
            self._add_style_influences()
            
            # Add contextual relationships
            self._add_contextual_relationships()
            
            logger.info(f"✅ Knowledge graph built: {self.knowledge_graph.number_of_nodes()} nodes, {self.knowledge_graph.number_of_edges()} edges")
            
        except Exception as e:
            logger.error(f"❌ Error building knowledge graph: {str(e)}")
            raise
    
    def _add_concept_nodes(self):
        """Add fashion concept nodes to the graph"""
        node_count = 0
        
        # Add garment types
        for category, items in self.fashion_ontology['garment_types'].items():
            self.knowledge_graph.add_node(category, type='garment_category', level='category')
            node_count += 1
            
            for item in items:
                self.knowledge_graph.add_node(item, type='garment_item', category=category, level='item')
                self.knowledge_graph.add_edge(item, category, relationship='belongs_to')
                node_count += 1
        
        # Add style concepts
        for style, attributes in self.fashion_ontology['style_concepts'].items():
            self.knowledge_graph.add_node(style, type='style_concept', level='style')
            node_count += 1
            
            for attribute in attributes:
                self.knowledge_graph.add_node(attribute, type='style_attribute', parent_style=style, level='attribute')
                self.knowledge_graph.add_edge(attribute, style, relationship='describes')
                node_count += 1
        
        # Add colors
        for color_group, colors in self.fashion_ontology['color_relationships'].items():
            self.knowledge_graph.add_node(color_group, type='color_group', level='group')
            node_count += 1
            
            for color in colors:
                if not self.knowledge_graph.has_node(color):
                    self.knowledge_graph.add_node(color, type='color', color_group=color_group, level='color')
                    node_count += 1
                self.knowledge_graph.add_edge(color, color_group, relationship='belongs_to')
        
        logger.info(f"   Added {node_count} concept nodes")
    
    def _add_compatibility_edges(self):
        """Add compatibility relationships"""
        edge_count = 0
        
        # Color compatibility
        complementary_pairs = [
            ('red', 'green'), ('blue', 'orange'), ('yellow', 'purple'),
            ('navy', 'coral'), ('black', 'white')
        ]
        
        for color1, color2 in complementary_pairs:
            if self.knowledge_graph.has_node(color1) and self.knowledge_graph.has_node(color2):
                self.knowledge_graph.add_edge(color1, color2, relationship='complementary', compatibility_score=0.9)
                edge_count += 1
        
        # Style compatibility
        style_pairs = [
            ('classic', 'elegant'), ('casual', 'comfortable'),
            ('formal', 'professional'), ('minimalist', 'modern')
        ]
        
        for style1, style2 in style_pairs:
            if self.knowledge_graph.has_node(style1) and self.knowledge_graph.has_node(style2):
                self.knowledge_graph.add_edge(style1, style2, relationship='highly_compatible', compatibility_score=0.9)
                edge_count += 1
        
        # Garment combinations
        classic_pairs = [
            ('blazer', 'trousers'), ('sweater', 'jeans'), ('dress', 'heels')
        ]
        
        for item1, item2 in classic_pairs:
            # Find matching nodes
            matching_nodes1 = [n for n in self.knowledge_graph.nodes() if item1 in n]
            matching_nodes2 = [n for n in self.knowledge_graph.nodes() if item2 in n]
            
            for node1 in matching_nodes1:
                for node2 in matching_nodes2:
                    self.knowledge_graph.add_edge(node1, node2, relationship='classic_pairing', compatibility_score=0.85)
                    edge_count += 1
        
        logger.info(f"   Added {edge_count} compatibility edges")
    
    def _add_semantic_relationships(self):
        """Add semantic relationships between concepts"""
        edge_count = 0
        
        # Similar styles
        similar_styles = [
            ('classic', 'elegant', 0.8),
            ('casual', 'comfortable', 0.9),
            ('formal', 'professional', 0.85),
            ('bohemian', 'artistic', 0.8),
            ('minimalist', 'modern', 0.75)
        ]
        
        for style1, style2, similarity in similar_styles:
            if self.knowledge_graph.has_node(style1) and self.knowledge_graph.has_node(style2):
                self.knowledge_graph.add_edge(style1, style2, relationship='semantically_similar', similarity_score=similarity)
                edge_count += 1
        
        logger.info(f"   Added {edge_count} semantic relationships")
    
    def _add_style_influences(self):
        """Add style influence networks"""
        edge_count = 0
        
        style_influences = {
            'classic': {'influenced_by': ['audrey_hepburn', 'grace_kelly'], 'influences': ['timeless_elegance']},
            'bohemian': {'influenced_by': ['1960s_hippie', 'folk_culture'], 'influences': ['free_spirit']},
            'minimalist': {'influenced_by': ['scandinavian_design'], 'influences': ['clean_lines']}
        }
        
        for style, influence_data in style_influences.items():
            if self.knowledge_graph.has_node(style):
                for source in influence_data['influenced_by']:
                    if not self.knowledge_graph.has_node(source):
                        self.knowledge_graph.add_node(source, type='influence_source', level='cultural')
                    self.knowledge_graph.add_edge(source, style, relationship='influences', influence_strength=0.8)
                    edge_count += 1
        
        logger.info(f"   Added {edge_count} influence relationships")
    
    def _add_contextual_relationships(self):
        """Add occasion and seasonal context relationships"""
        edge_count = 0
        
        # Seasonal appropriateness
        seasonal_items = {
            'spring': ['light_jacket', 'cardigan'],
            'summer': ['shorts', 't-shirts', 'sandals'],
            'fall': ['sweater', 'boots'],
            'winter': ['coat', 'scarf']
        }
        
        for season, items in seasonal_items.items():
            if not self.knowledge_graph.has_node(season):
                self.knowledge_graph.add_node(season, type='season', level='context')
            
            for item in items:
                matching_nodes = [n for n in self.knowledge_graph.nodes() if item.lower() in n.lower()]
                for node in matching_nodes:
                    self.knowledge_graph.add_edge(node, season, relationship='appropriate_for', appropriateness_score=0.8)
                    edge_count += 1
        
        logger.info(f"   Added {edge_count} contextual relationships")
    
    def query(self, query_type, params):
        """Query the fashion knowledge graph"""
        try:
            if query_type == 'compatibility':
                return self._query_compatibility(params)
            elif query_type == 'recommendations':
                return self._query_recommendations(params)
            elif query_type == 'semantic_search':
                return self._query_semantic_search(params)
            elif query_type == 'style_analysis':
                return self._query_style_analysis(params)
            else:
                raise ValueError(f"Unknown query type: {query_type}")
                
        except Exception as e:
            logger.error(f"Knowledge graph query error: {str(e)}")
            raise
    
    def _query_compatibility(self, params):
        """Query compatibility between fashion items"""
        item1 = params.get('item1')
        item2 = params.get('item2')
        context = params.get('context', 'general')
        
        if not item1 or not item2:
            return {'compatible': False, 'score': 0.0, 'reasoning': 'Missing items'}
        
        # Find nodes in knowledge graph
        item1_nodes = self._find_matching_nodes(item1)
        item2_nodes = self._find_matching_nodes(item2)
        
        if not item1_nodes or not item2_nodes:
            return {'compatible': False, 'score': 0.0, 'reasoning': 'Items not found in knowledge graph'}
        
        # Calculate compatibility scores
        best_score = 0.0
        best_reasoning = ""
        
        for node1 in item1_nodes:
            for node2 in item2_nodes:
                score = self._calculate_node_compatibility(node1, node2, context)
                if score > best_score:
                    best_score = score
                    best_reasoning = self._generate_compatibility_reasoning(node1, node2, score)
        
        return {
            'compatible': best_score > 0.6,
            'score': best_score,
            'reasoning': best_reasoning,
            'context_appropriate': context != 'general'
        }
    
    def _query_recommendations(self, params):
        """Query style recommendations"""
        base_item = params.get('base_item')
        context = params.get('context', 'general')
        limit = params.get('limit', 5)
        
        if not base_item:
            return {'recommendations': [], 'reasoning': 'No base item provided'}
        
        base_nodes = self._find_matching_nodes(base_item)
        if not base_nodes:
            return {'recommendations': [], 'reasoning': 'Base item not found in knowledge graph'}
        
        recommendations = []
        for base_node in base_nodes:
            compatible_items = self._traverse_for_recommendations(base_node, context)
            recommendations.extend(compatible_items)
        
        # Sort by compatibility score and remove duplicates
        unique_recommendations = {}
        for rec in recommendations:
            item = rec['recommended_item']
            if item not in unique_recommendations or rec['compatibility_score'] > unique_recommendations[item]['compatibility_score']:
                unique_recommendations[item] = rec
        
        sorted_recommendations = sorted(unique_recommendations.values(), key=lambda x: x['compatibility_score'], reverse=True)
        
        return {
            'recommendations': sorted_recommendations[:limit],
            'total_found': len(sorted_recommendations),
            'base_item': base_item
        }
    
    def _query_semantic_search(self, params):
        """Perform semantic search in the knowledge graph"""
        query_term = params.get('query', '')
        search_type = params.get('type', 'all')
        limit = params.get('limit', 10)
        
        if not query_term:
            return {'results': [], 'reasoning': 'No query term provided'}
        
        # Find semantically related nodes
        related_nodes = []
        query_lower = query_term.lower()
        
        for node in self.knowledge_graph.nodes(data=True):
            node_name, node_data = node
            
            # Direct match
            if query_lower in node_name.lower():
                related_nodes.append({
                    'node': node_name,
                    'type': node_data.get('type', 'unknown'),
                    'relevance': 1.0,
                    'match_type': 'direct'
                })
            
            # Semantic similarity (check connected nodes)
            for neighbor in self.knowledge_graph.neighbors(node_name):
                if query_lower in neighbor.lower():
                    edge_data = self.knowledge_graph.get_edge_data(node_name, neighbor)
                    if edge_data:
                        relevance = 0.7  # Base relevance for connected nodes
                        related_nodes.append({
                            'node': node_name,
                            'type': node_data.get('type', 'unknown'),
                            'relevance': relevance,
                            'match_type': 'semantic',
                            'connection': neighbor
                        })
        
        # Sort by relevance and limit results
        related_nodes.sort(key=lambda x: x['relevance'], reverse=True)
        
        return {
            'results': related_nodes[:limit],
            'total_found': len(related_nodes),
            'query': query_term
        }
    
    def _query_style_analysis(self, params):
        """Analyze style profile based on items"""
        items = params.get('items', [])
        
        if not items:
            return {'style_profile': {}, 'reasoning': 'No items provided'}
        
        # Map items to knowledge graph concepts
        style_concepts = defaultdict(int)
        garment_categories = defaultdict(int)
        color_preferences = defaultdict(int)
        
        for item in items:
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
        
        return {
            'style_profile': profile,
            'total_items_analyzed': len(items),
            'reasoning': f'Analyzed {len(items)} items to determine style preferences'
        }
    
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
    
    def _calculate_node_compatibility(self, node1, node2, context):
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
            common_neighbors = set(self.knowledge_graph.neighbors(node1)) & set(self.knowledge_graph.neighbors(node2))
            if common_neighbors:
                shared_score = len(common_neighbors) / 10.0  # Normalize
                score = min(shared_score, 0.7)  # Cap at 0.7 for indirect
        
        return score
    
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
    
    def _traverse_for_recommendations(self, base_node, context):
        """Traverse knowledge graph to find recommendations"""
        recommendations = []
        
        # Get direct neighbors with compatibility relationships
        for neighbor in self.knowledge_graph.neighbors(base_node):
            edge_data = self.knowledge_graph.get_edge_data(base_node, neighbor)
            
            for edge in edge_data.values():
                relationship = edge.get('relationship', '')
                
                if relationship in ['highly_compatible', 'classic_pairing', 'moderately_compatible']:
                    score = edge.get('compatibility_score', 0.5)
                    
                    recommendations.append({
                        'recommended_item': neighbor,
                        'compatibility_score': score,
                        'reasoning': f"Recommended based on {relationship} with {base_node}"
                    })
        
        return recommendations
    
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
    
    def save_knowledge_graph(self):
        """Save the knowledge graph to file"""
        try:
            with open(self.knowledge_graph_path, 'wb') as f:
                pickle.dump(self.knowledge_graph, f)
            logger.info(f"✅ Knowledge graph saved: {self.knowledge_graph_path}")
        except Exception as e:
            logger.error(f"❌ Error saving knowledge graph: {str(e)}")
    
    def is_loaded(self):
        """Check if knowledge graph is loaded"""
        return self.knowledge_graph.number_of_nodes() > 0
    
    def get_statistics(self):
        """Get knowledge graph statistics"""
        return {
            'nodes': self.knowledge_graph.number_of_nodes(),
            'edges': self.knowledge_graph.number_of_edges(),
            'node_types': list(set(data.get('type', 'unknown') for _, data in self.knowledge_graph.nodes(data=True))),
            'relationship_types': list(set(data.get('relationship', 'unknown') for _, _, data in self.knowledge_graph.edges(data=True))),
            'ontology_categories': list(self.fashion_ontology.keys()),
            'is_loaded': self.is_loaded()
        }
