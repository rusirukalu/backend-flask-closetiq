# services/compatibility_service.py - Advanced Style Compatibility Engine
import numpy as np
import json
import os
from datetime import datetime
import logging
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class StyleCompatibilityEngine:
    """Advanced Fashion Style Compatibility Analysis"""
    
    def __init__(self):
        self.compatibility_rules = self.load_compatibility_rules()
        self.color_harmony_matrix = self.build_color_harmony_matrix()
        self.style_compatibility_matrix = self.build_style_compatibility_matrix()
        
        # Load fashion knowledge
        self.fashion_rules = self.load_fashion_rules()
        
    def load_compatibility_rules(self):
        """Load comprehensive compatibility rules"""
        return {
            'color_harmony': {
                'complementary_pairs': [
                    ('red', 'green'), ('blue', 'orange'), ('yellow', 'purple'),
                    ('navy', 'coral'), ('forest_green', 'burgundy')
                ],
                'analogous_groups': [
                    ['blue', 'blue_green', 'green'],
                    ['red', 'red_orange', 'orange'],
                    ['yellow', 'yellow_green', 'green']
                ],
                'neutral_pairs': [
                    ('black', 'white'), ('gray', 'white'), ('navy', 'cream'),
                    ('brown', 'beige'), ('charcoal', 'ivory')
                ],
                'monochromatic_families': {
                    'blue': ['navy', 'royal_blue', 'sky_blue', 'powder_blue'],
                    'gray': ['charcoal', 'slate', 'silver', 'light_gray'],
                    'green': ['forest_green', 'olive', 'mint', 'sage']
                }
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
                'layering_compatible': [
                    ('tank', 'cardigan'), ('shirt', 'blazer'),
                    ('dress', 'jacket'), ('tshirt', 'vest')
                ],
                'avoid_combinations': [
                    ('formal_shoes', 'athletic_wear'),
                    ('evening_gown', 'sneakers'),
                    ('business_suit', 'flip_flops')
                ]
            },
            'seasonal_compatibility': {
                'spring': ['light_layers', 'pastels', 'breathable_fabrics'],
                'summer': ['lightweight', 'bright_colors', 'minimal_layers'],
                'fall': ['layers', 'earth_tones', 'medium_weight'],
                'winter': ['heavy_fabrics', 'dark_colors', 'full_coverage']
            }
        }
    
    def build_color_harmony_matrix(self):
        """Build color harmony compatibility matrix"""
        colors = ['black', 'white', 'red', 'blue', 'green', 'yellow', 'pink', 'purple', 'brown', 'gray', 'orange', 'navy', 'beige']
        
        # Initialize matrix
        matrix = np.zeros((len(colors), len(colors)))
        
        # Fill matrix based on compatibility rules
        for i, color1 in enumerate(colors):
            for j, color2 in enumerate(colors):
                score = self.calculate_color_compatibility(color1, color2)
                matrix[i][j] = score
        
        return {
            'matrix': matrix,
            'color_indices': {color: i for i, color in enumerate(colors)},
            'colors': colors
        }
    
    def calculate_color_compatibility(self, color1, color2):
        """Calculate compatibility score between two colors"""
        # Same color
        if color1 == color2:
            return 0.5  # Neutral - can work but not interesting
        
        # Check complementary pairs
        complementary_pairs = self.compatibility_rules['color_harmony']['complementary_pairs']
        if (color1, color2) in complementary_pairs or (color2, color1) in complementary_pairs:
            return 0.95
        
        # Check neutral pairs
        neutral_pairs = self.compatibility_rules['color_harmony']['neutral_pairs']
        if (color1, color2) in neutral_pairs or (color2, color1) in neutral_pairs:
            return 0.90
        
        # Check if both are neutrals
        neutrals = ['black', 'white', 'gray', 'beige', 'brown', 'navy']
        if color1 in neutrals and color2 in neutrals:
            return 0.85
        
        # Check monochromatic families
        monochromatic = self.compatibility_rules['color_harmony']['monochromatic_families']
        for family_colors in monochromatic.values():
            if color1 in family_colors and color2 in family_colors:
                return 0.80
        
        # Check analogous colors (simplified)
        analogous_groups = self.compatibility_rules['color_harmony']['analogous_groups']
        for group in analogous_groups:
            if color1 in group and color2 in group:
                return 0.75
        
        # Default compatibility
        return 0.60
    
    def build_style_compatibility_matrix(self):
        """Build style compatibility matrix"""
        styles = ['classic', 'casual', 'formal', 'trendy', 'bohemian', 'minimalist', 'edgy', 'romantic']
        
        # Initialize matrix
        matrix = np.zeros((len(styles), len(styles)))
        
        # Fill matrix based on style compatibility rules
        for i, style1 in enumerate(styles):
            for j, style2 in enumerate(styles):
                score = self.calculate_style_compatibility(style1, style2)
                matrix[i][j] = score
        
        return {
            'matrix': matrix,
            'style_indices': {style: i for i, style in enumerate(styles)},
            'styles': styles
        }
    
    def calculate_style_compatibility(self, style1, style2):
        """Calculate compatibility score between two styles"""
        # Same style
        if style1 == style2:
            return 1.0
        
        # Check high compatibility
        high_compat = self.compatibility_rules['style_compatibility']['high_compatibility']
        if (style1, style2) in high_compat or (style2, style1) in high_compat:
            return 0.90
        
        # Check moderate compatibility
        moderate_compat = self.compatibility_rules['style_compatibility']['moderate_compatibility']
        if (style1, style2) in moderate_compat or (style2, style1) in moderate_compat:
            return 0.70
        
        # Check style conflicts
        conflicts = self.compatibility_rules['style_compatibility']['style_conflicts']
        if (style1, style2) in conflicts or (style2, style1) in conflicts:
            return 0.20
        
        # Default moderate compatibility
        return 0.60
    
    def load_fashion_rules(self):
        """Load fashion rules and guidelines"""
        return {
            'proportion_rules': {
                'fitted_top_loose_bottom': 0.9,
                'loose_top_fitted_bottom': 0.9,
                'avoid_baggy_on_baggy': 0.1
            },
            'color_rules': {
                'max_colors': 3,
                'neutral_base_bonus': 0.1,
                'metallic_mixing_penalty': -0.2
            },
            'occasion_appropriateness': {
                'work': ['formal', 'business', 'conservative'],
                'party': ['trendy', 'bold', 'statement'],
                'casual': ['comfortable', 'relaxed', 'everyday'],
                'formal': ['elegant', 'sophisticated', 'polished']
            }
        }
    
    def check_compatibility(self, item1_id, item2_id, context='general'):
        """Check compatibility between two fashion items"""
        try:
            # For this demo, we'll simulate item data
            # In production, you'd fetch actual item data from the database
            item1_data = self.get_item_data(item1_id)
            item2_data = self.get_item_data(item2_id)
            
            if not item1_data or not item2_data:
                raise Exception("Item data not found")
            
            # Calculate different aspects of compatibility
            color_score = self.calculate_color_compatibility_score(item1_data, item2_data)
            style_score = self.calculate_style_compatibility_score(item1_data, item2_data)
            garment_score = self.calculate_garment_compatibility_score(item1_data, item2_data)
            context_score = self.calculate_context_compatibility_score(item1_data, item2_data, context)
            
            # Calculate overall compatibility
            weights = {
                'color': 0.3,
                'style': 0.3,
                'garment': 0.25,
                'context': 0.15
            }
            
            overall_score = (
                color_score * weights['color'] +
                style_score * weights['style'] +
                garment_score * weights['garment'] +
                context_score * weights['context']
            )
            
            # Generate reasoning
            reasoning = self.generate_compatibility_reasoning(
                item1_data, item2_data, color_score, style_score, garment_score, context_score
            )
            
            # Generate recommendations
            recommendations = self.generate_improvement_recommendations(
                item1_data, item2_data, color_score, style_score, garment_score
            )
            
            return {
                'score': float(overall_score),
                'breakdown': {
                    'color_compatibility': float(color_score),
                    'style_compatibility': float(style_score),
                    'garment_compatibility': float(garment_score),
                    'context_compatibility': float(context_score)
                },
                'reasoning': reasoning,
                'recommendations': recommendations,
                'compatible': overall_score > 0.6,
                'confidence': min(0.95, overall_score + 0.1)
            }
            
        except Exception as e:
            logger.error(f"Compatibility check error: {str(e)}")
            raise
    
    def get_item_data(self, item_id):
        """Get item data (simulated for demo)"""
        # In production, this would fetch from the database
        # For demo, we'll return simulated data
        simulated_items = {
            'item1': {
                'category': 'shirts_blouses',
                'colors': ['white', 'blue'],
                'style': 'classic',
                'patterns': ['solid'],
                'formality': 'business'
            },
            'item2': {
                'category': 'pants_jeans',
                'colors': ['navy', 'blue'],
                'style': 'casual',
                'patterns': ['solid'],
                'formality': 'casual'
            }
        }
        
        return simulated_items.get(item_id, {
            'category': 'unknown',
            'colors': ['unknown'],
            'style': 'casual',
            'patterns': ['unknown'],
            'formality': 'casual'
        })
    
    def calculate_color_compatibility_score(self, item1, item2):
        """Calculate color compatibility between two items"""
        try:
            item1_colors = item1.get('colors', [])
            item2_colors = item2.get('colors', [])
            
            if not item1_colors or not item2_colors:
                return 0.5  # Neutral if no color data
            
            # Calculate best color pairing
            best_score = 0
            for color1 in item1_colors:
                for color2 in item2_colors:
                    score = self.calculate_color_compatibility(color1, color2)
                    best_score = max(best_score, score)
            
            return best_score
            
        except Exception as e:
            logger.error(f"Color compatibility calculation error: {str(e)}")
            return 0.5
    
    def calculate_style_compatibility_score(self, item1, item2):
        """Calculate style compatibility between two items"""
        try:
            style1 = item1.get('style', 'casual')
            style2 = item2.get('style', 'casual')
            
            return self.calculate_style_compatibility(style1, style2)
            
        except Exception as e:
            logger.error(f"Style compatibility calculation error: {str(e)}")
            return 0.5
    
    def calculate_garment_compatibility_score(self, item1, item2):
        """Calculate garment type compatibility"""
        try:
            category1 = item1.get('category', 'unknown')
            category2 = item2.get('category', 'unknown')
            
            # Check classic pairs
            classic_pairs = self.compatibility_rules['garment_combinations']['classic_pairs']
            for pair in classic_pairs:
                if (category1 in pair[0] and category2 in pair[1]) or \
                   (category1 in pair[1] and category2 in pair[0]):
                    return 0.9
            
            # Check layering compatibility
            layering = self.compatibility_rules['garment_combinations']['layering_compatible']
            for pair in layering:
                if (category1 in pair[0] and category2 in pair[1]) or \
                   (category1 in pair[1] and category2 in pair[0]):
                    return 0.8
            
            # Check avoid combinations
            avoid = self.compatibility_rules['garment_combinations']['avoid_combinations']
            for pair in avoid:
                if (category1 in pair[0] and category2 in pair[1]) or \
                   (category1 in pair[1] and category2 in pair[0]):
                    return 0.2
            
            # Default moderate compatibility
            return 0.65
            
        except Exception as e:
            logger.error(f"Garment compatibility calculation error: {str(e)}")
            return 0.5
    
    def calculate_context_compatibility_score(self, item1, item2, context):
        """Calculate context-based compatibility"""
        try:
            formality1 = item1.get('formality', 'casual')
            formality2 = item2.get('formality', 'casual')
            
            # Check if both items are appropriate for the context
            context_mapping = {
                'work': ['business', 'formal'],
                'party': ['trendy', 'formal', 'bold'],
                'casual': ['casual', 'comfortable'],
                'formal': ['formal', 'business', 'elegant']
            }
            
            appropriate_styles = context_mapping.get(context, ['casual'])
            
            item1_appropriate = formality1 in appropriate_styles
            item2_appropriate = formality2 in appropriate_styles
            
            if item1_appropriate and item2_appropriate:
                return 0.9
            elif item1_appropriate or item2_appropriate:
                return 0.6
            else:
                return 0.3
                
        except Exception as e:
            logger.error(f"Context compatibility calculation error: {str(e)}")
            return 0.5
    
    def generate_compatibility_reasoning(self, item1, item2, color_score, style_score, garment_score, context_score):
        """Generate human-readable reasoning for compatibility"""
        reasoning_parts = []
        
        # Color reasoning
        if color_score > 0.8:
            reasoning_parts.append("Colors work beautifully together")
        elif color_score > 0.6:
            reasoning_parts.append("Colors are compatible")
        else:
            reasoning_parts.append("Color combination could be improved")
        
        # Style reasoning
        if style_score > 0.8:
            reasoning_parts.append("Styles complement each other well")
        elif style_score > 0.6:
            reasoning_parts.append("Styles are moderately compatible")
        else:
            reasoning_parts.append("Style mismatch detected")
        
        # Garment reasoning
        if garment_score > 0.8:
            reasoning_parts.append("Classic garment pairing")
        elif garment_score > 0.6:
            reasoning_parts.append("Good garment combination")
        else:
            reasoning_parts.append("Unconventional garment pairing")
        
        return ". ".join(reasoning_parts) + "."
    
    def generate_improvement_recommendations(self, item1, item2, color_score, style_score, garment_score):
        """Generate recommendations for improving compatibility"""
        recommendations = []
        
        if color_score < 0.6:
            recommendations.append("Consider adding a neutral accessory to bridge the color gap")
        
        if style_score < 0.6:
            recommendations.append("Try styling with accessories that match both pieces")
        
        if garment_score < 0.6:
            recommendations.append("Consider adding a third piece to create better harmony")
        
        if not recommendations:
            recommendations.append("This combination works well as is!")
        
        return recommendations
    
    def batch_compatibility_check(self, item_pairs, context='general'):
        """Check compatibility for multiple item pairs"""
        try:
            results = []
            
            for pair in item_pairs:
                item1_id, item2_id = pair
                try:
                    compatibility = self.check_compatibility(item1_id, item2_id, context)
                    results.append({
                        'item1_id': item1_id,
                        'item2_id': item2_id,
                        'compatibility': compatibility,
                        'success': True
                    })
                except Exception as e:
                    results.append({
                        'item1_id': item1_id,
                        'item2_id': item2_id,
                        'success': False,
                        'error': str(e)
                    })
            
            return {
                'results': results,
                'total_pairs': len(item_pairs),
                'successful_checks': len([r for r in results if r['success']])
            }
            
        except Exception as e:
            logger.error(f"Batch compatibility check error: {str(e)}")
            raise
    
    def get_compatibility_statistics(self):
        """Get compatibility engine statistics"""
        return {
            'color_matrix_size': self.color_harmony_matrix['matrix'].shape,
            'style_matrix_size': self.style_compatibility_matrix['matrix'].shape,
            'supported_colors': len(self.color_harmony_matrix['colors']),
            'supported_styles': len(self.style_compatibility_matrix['styles']),
            'rule_categories': list(self.compatibility_rules.keys()),
            'fashion_rules_loaded': len(self.fashion_rules)
        }
