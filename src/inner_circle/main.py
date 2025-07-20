import pygame
import math
import random

from datetime import datetime
import cv2
import numpy as np

DEBUG = False  # Set to True for debug output

# Video recording class
class VideoRecorder:
    def __init__(self, filename=None, fps=60, width=405, height=720):
        """
        Initialize video recorder.
        
        Args:
            filename (str): Output filename (auto-generated if None)
            fps (int): Frames per second
            width, height (int): Video dimensions
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ball_escape_{timestamp}.mp4"
        
        self.filename = filename
        self.fps = fps
        self.width = width
        self.height = height
        self.recording = False
        self.writer = None
        self.frame_count = 0
        
    def start_recording(self):
        """Start video recording."""
        if self.recording:
            return False
        
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(self.filename, fourcc, self.fps, (self.width, self.height))
        
        if not self.writer.isOpened():
            print(f"Error: Could not open video writer for {self.filename}")
            return False
        
        self.recording = True
        self.frame_count = 0
        print(f"Started recording: {self.filename}")
        return True
    
    def stop_recording(self):
        """Stop video recording."""
        if not self.recording:
            return False
        
        self.recording = False
        if self.writer:
            self.writer.release()
            self.writer = None
        
        print(f"Stopped recording. Saved {self.frame_count} frames to {self.filename}")
        return True
    
    def capture_frame(self, surface):
        """
        Capture a frame from pygame surface.
        
        Args:
            surface: Pygame surface to capture
        """
        if not self.recording or not self.writer:
            return
        
        # Convert pygame surface to numpy array
        frame_array = pygame.surfarray.array3d(surface)
        
        # Pygame uses (width, height, 3) format, OpenCV uses (height, width, 3)
        frame_array = np.transpose(frame_array, (1, 0, 2))
        
        # Convert RGB to BGR (OpenCV format)
        frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
        
        # Write frame to video
        self.writer.write(frame_bgr)
        self.frame_count += 1
    
    def is_recording(self):
        """Check if currently recording."""
        return self.recording

class Ball:
    def __init__(self, x, y, radius=8, color=(255, 255, 0)):
        """
        Initialize a ball with physics.
        
        Args:
            x, y (float): Starting position
            radius (int): Ball radius in pixels
            color (tuple): RGB color tuple
        """
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
        
        # Random velocity in a random direction (scaled appropriately)
        speed = 60 #random.uniform(30, 60)  # pixels per second (reduced speed)
        angle = random.uniform(0, 2 * math.pi)
        self.vx = speed * math.cos(angle)
        self.vy = speed * math.sin(angle)
        
        # Physics constants
        self.gravity = 35  # pixels per second squared (reduced gravity)
        self.bounce_damping = 0.1  # Energy retention on screen bounces
        self.air_resistance = 0.998  # Very slight air resistance per second
    
    def update(self, dt, screen_width, screen_height, arcs):
        """
        Update ball position and handle collisions.
        
        Args:
            dt (float): Time delta in seconds
            screen_width, screen_height (int): Screen dimensions
            arcs (list): List of RotatingArc objects for collision detection
        """        
        # Apply physics forces over time
        # Gravity acceleration (constant downward force)
        self.vy += self.gravity * dt
        
        # Apply air resistance (exponential decay based on dt)
        resistance_factor = math.pow(self.air_resistance, dt)
        self.vx *= resistance_factor
        self.vy *= resistance_factor
        
        # Update position based on velocity and time
        self.x += self.vx * dt
        self.y += self.vy * dt
        
        # Optimized collision detection - check all arcs in one pass
        self._check_arc_collisions(arcs)

    def draw(self, surface):
        """Draw the ball with anti-aliasing."""
        # Draw main ball
        pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), self.radius)
        # Add a highlight for 3D effect
        highlight_color = tuple(min(255, c + 80) for c in self.color)
        pygame.draw.circle(surface, highlight_color, 
                         (int(self.x - self.radius//3), int(self.y - self.radius//3)), 
                         max(1, self.radius//3))
    
    def get_distance_from_point(self, px, py):
        """Calculate distance from ball center to a point."""
        return math.sqrt((self.x - px)**2 + (self.y - py)**2)
    
    def _check_arc_collisions(self, arcs):
        """
        Collision detection - check all arcs in one pass.
        
        Args:
            arcs (list): List of RotatingArc objects
        """
        if not arcs:
            return
        
        # Get the center point (assuming all arcs share the same center)
        center = arcs[0].center
        
        # Calculate ball's position relative to center
        center_to_ball_x = self.x - center[0]
        center_to_ball_y = self.y - center[1]
        distance = math.sqrt(center_to_ball_x**2 + center_to_ball_y**2)
        
        if distance == 0:
            return  # Ball is exactly at center, no collision possible
        
        # Calculate ball's angle from center
        ball_angle = math.atan2(center_to_ball_y, center_to_ball_x)
        
        # Normalize direction vector
        normal_x = center_to_ball_x / distance
        normal_y = center_to_ball_y / distance
        
        # Find which arcs could potentially collide with the ball
        collision_candidates = []
        for arc in arcs:
            if not arc.active:
                continue
                
            # Check if ball is within collision range of this arc
            inner_radius = arc.radius
            outer_radius = arc.radius + arc.width
            
            if (inner_radius - self.radius <= distance <= outer_radius + self.radius):
                collision_candidates.append(arc)
        
        if not collision_candidates:
            return  # No arcs in collision range
        
        # Sort candidates by radius (innermost first) to prioritize inner collisions
        collision_candidates.sort(key=lambda arc: arc.radius)
        
        # Check if the ball's angle intersects with any solid part of the candidate arcs
        # We need to check ALL arcs, not just the first one, because gaps might overlap
        colliding_arc = None
        collision_type = None
        
        for arc in collision_candidates:
            # Check if ball is in a gap for this arc
            if arc._is_angle_in_gap(ball_angle):
                continue  # Ball is in gap, no collision with this arc
            
            # Ball is hitting solid part of this arc
            inner_radius = arc.radius
            outer_radius = arc.radius + arc.width
            
            if distance > arc.radius:
                # Hitting outer edge - prioritize innermost arc for outer collisions
                if colliding_arc is None:
                    colliding_arc = arc
                    collision_type = 'outer'
                    target_distance = outer_radius + self.radius + 2
                    collision_normal_x = normal_x
                    collision_normal_y = normal_y
            else:
                # Hitting inner edge - this takes priority over outer edge collisions
                colliding_arc = arc
                collision_type = 'inner'
                target_distance = inner_radius - self.radius - 2
                collision_normal_x = -normal_x
                collision_normal_y = -normal_y
                arc.ball_was_inside = True
                break  # Inner collision takes absolute priority
        
        if not colliding_arc:
            return  # No actual collision occurred
        
        # Apply collision response
        # Correct ball position to prevent overlap
        self.x = center[0] + normal_x * target_distance
        self.y = center[1] + normal_y * target_distance
        
        # Apply reflection using the collision normal
        dot_product = self.vx * collision_normal_x + self.vy * collision_normal_y
        self.vx = self.vx - 2 * dot_product * collision_normal_x
        self.vy = self.vy - 2 * dot_product * collision_normal_y


class RotatingArc:
    def __init__(self, center, radius, width=5, color=(255, 255, 255), 
                 gap_angle=90, rotation_speed=180, segments=150):
        """
        Initialize a rotating arc with anti-aliasing.
        
        Args:
            center (tuple): (x, y) center coordinates
            radius (int): Arc radius in pixels
            width (int): Arc line width
            color (tuple): RGB color tuple
            gap_angle (float): Gap size in degrees (0-360)
            rotation_speed (float): Rotation speed in degrees per second
            segments (int): Number of line segments to draw (more = smoother)
        """
        self.center = center
        self.radius = radius
        self.width = width
        self.color = color
        self.gap_angle = math.radians(gap_angle)  # Convert to radians
        self.rotation_speed = math.radians(rotation_speed)  # Convert to radians/sec
        self.current_rotation = 0.0  # Current rotation angle in radians
        self.segments = segments
        self.active = True  # Whether this arc should be drawn/updated
        
        # Calculate arc span (full circle minus gap)
        self.arc_span = 2 * math.pi - self.gap_angle
        
        # Track if ball has been inside this arc
        self.ball_was_inside = True  # Start as True since ball starts at center
    
    def update(self, dt):
        """
        Update the arc rotation based on time delta.
        
        Args:
            dt (float): Time delta in seconds
        """
        if not self.active:
            return
            
        self.current_rotation += self.rotation_speed * dt
        # Keep rotation within 0-2π range
        self.current_rotation %= (2 * math.pi)
    
    def _get_point_on_circle(self, angle):
        """Calculate point on circle at given angle."""
        x = self.center[0] + self.radius * math.cos(angle)
        y = self.center[1] + self.radius * math.sin(angle)
        return (x, y)
    
    def _get_point_on_circle_radius(self, angle, radius):
        """Calculate point on circle at given angle and radius."""
        x = self.center[0] + radius * math.cos(angle)
        y = self.center[1] + radius * math.sin(angle)
        return (x, y)
    
    def _normalize_angle(self, angle):
        """Normalize angle to 0-2π range."""
        return angle % (2 * math.pi)
    
    def _is_angle_in_gap(self, angle):
        """Check if an angle is within the gap."""
        angle = self._normalize_angle(angle)
        gap_start = self._normalize_angle(self.current_rotation + self.arc_span)
        gap_end = self._normalize_angle(self.current_rotation + 2 * math.pi)
        
        if gap_start < gap_end:
            return gap_start <= angle <= gap_end
        else:
            return angle >= gap_start or angle <= gap_end
    
    def check_ball_collision(self, ball):
        """
        Check if ball collides with the arc and handle bounce.
        
        Args:
            ball (Ball): The ball to check
            
        Returns:
            bool: True if collision occurred
        """
        if not self.active:
            return False
        
        # Calculate distance from ball center to arc center
        distance = ball.get_distance_from_point(self.center[0], self.center[1])
        
        # Define collision boundaries (arc thickness)
        inner_radius = self.radius
        outer_radius = self.radius + self.width
        
        # Update inside status more reliably
        if distance < inner_radius:
            self.ball_was_inside = True
        
        # Check if ball is within arc collision range
        collision_range = (inner_radius - ball.radius <= distance <= outer_radius + ball.radius)
        
        if not collision_range:
            return False
        
        # Calculate current ball angle
        ball_angle = math.atan2(ball.y - self.center[1], ball.x - self.center[0])
        
        # Check if ball is in the gap (no collision if in gap)
        if self._is_angle_in_gap(ball_angle):
            return False
        
        # Ball is hitting the solid part of the arc - proceed with collision
        
        # Calculate proper collision normal (radial direction from center)
        center_to_ball_x = ball.x - self.center[0]
        center_to_ball_y = ball.y - self.center[1]
        
        # Normalize the vector from center to ball
        if distance > 0:
            normal_x = center_to_ball_x / distance
            normal_y = center_to_ball_y / distance
        else:
            # Fallback if ball is exactly at center
            normal_x, normal_y = 1.0, 0.0
        
        # Determine collision type
        hitting_outer = distance > self.radius
        
        if hitting_outer:
            # Hitting outer edge - normal points outward (away from center)
            collision_normal_x = normal_x
            collision_normal_y = normal_y
            # Position ball outside the arc
            target_distance = outer_radius + ball.radius + 2
        else:
            # Hitting inner edge - normal points inward (toward center)
            collision_normal_x = -normal_x
            collision_normal_y = -normal_y
            # Position ball inside the arc
            target_distance = inner_radius - ball.radius - 2
            self.ball_was_inside = True
        
        # Correct ball position to prevent overlap
        ball.x = self.center[0] + normal_x * target_distance
        ball.y = self.center[1] + normal_y * target_distance
        
        # Store original speed before reflection
        original_speed = math.sqrt(ball.vx**2 + ball.vy**2)
        
        # Apply reflection using the collision normal
        # Formula: v_reflected = v - 2 * (v · n) * n
        dot_product = ball.vx * collision_normal_x + ball.vy * collision_normal_y
        
        # Apply reflection
        ball.vx = ball.vx - 2 * dot_product * collision_normal_x
        ball.vy = ball.vy - 2 * dot_product * collision_normal_y
        
        # Don't normalize velocity - let gravity and natural physics take over
        # This allows for more realistic, non-perfect bounces
        
        return True
    
    def check_ball_escaped(self, ball: Ball):
        """
        Check if the ball has escaped by touching the inner edge of the gap.
        
        Args:
            ball (Ball): The ball to check
            
        Returns:
            bool: True if ball escaped through the gap
        """

        if not self.active or not self.ball_was_inside:
            return False
        
        # Calculate ball's angle from center
        ball_angle = math.atan2(ball.y - self.center[1], ball.x - self.center[0])
        
        # Check if ball is currently in the gap
        if not self._is_angle_in_gap(ball_angle):
            return False  # Ball is not in gap area
        
        # Calculate distance from ball to arc center
        distance = ball.get_distance_from_point(self.center[0], self.center[1])
        
        # Check if ball has reached the inner edge of the arc while in the gap
        # This means the ball is trying to exit through the gap from inside
        if distance >= self.radius - ball.radius:
            # Ball has touched the inner edge of the gap - it escaped!
            print(f"Ball escaped through arc at radius {self.radius}!")  # Debug
            self.active = False  # Deactivate this arc
            return True
        
        return False
    
    def draw(self, surface):
        """
        Draw the anti-aliased rotating arc on the given surface.
        
        Args:
            surface: Pygame surface to draw on
        """
        if not self.active or self.arc_span <= 0:
            return
        
        # Calculate the angle step between segments
        angle_step = self.arc_span / self.segments
        
        # For thicker lines, we'll draw multiple concentric arcs
        thickness_layers = max(1, self.width // 2)
        
        for layer in range(thickness_layers):
            # Calculate radius for this layer (creates thickness effect)
            layer_radius_offset = (layer - thickness_layers // 2) * 0.5
            current_radius = self.radius + layer_radius_offset
            
            # Calculate alpha for smoother thickness gradient
            alpha = 1.0
            if thickness_layers > 1:
                alpha = 1.0 - (abs(layer - thickness_layers // 2) / thickness_layers) * 0.3
            
            # Apply alpha to color
            layer_color = tuple(int(c * alpha) for c in self.color)
            
            # Draw the arc as connected line segments
            points = []
            for i in range(self.segments + 1):
                angle = self.current_rotation + (i * angle_step)
                x = self.center[0] + current_radius * math.cos(angle)
                y = self.center[1] + current_radius * math.sin(angle)
                points.append((x, y))
            
            # Draw anti-aliased lines between consecutive points
            for i in range(len(points) - 1):
                pygame.draw.aaline(surface, layer_color, points[i], points[i + 1])
        
        # For very thick lines, also draw filled circles at the endpoints for rounded caps
        # if self.width >= 4:
        #     start_angle = self.current_rotation
        #     end_angle = self.current_rotation + self.arc_span
            
        #     start_point = self._get_point_on_circle(start_angle)
        #     end_point = self._get_point_on_circle(end_angle)
            
        #     cap_radius = max(1, self.width // 2)
        #     pygame.draw.circle(surface, self.color, 
        #                      (int(start_point[0]), int(start_point[1])), cap_radius)
        #     pygame.draw.circle(surface, self.color, 
        #                      (int(end_point[0]), int(end_point[1])), cap_radius)

        # Debug: Draw gap boundaries
        if DEBUG:
            gap_start = self._normalize_angle(self.current_rotation + self.arc_span)
            gap_end = self._normalize_angle(self.current_rotation + 2 * math.pi)
            
            # Draw gap start line (red)
            gap_start_inner = self._get_point_on_circle_radius(gap_start, self.radius - self.width // 2 - 5)
            gap_start_outer = self._get_point_on_circle_radius(gap_start, self.radius + self.width // 2 + 5)
            pygame.draw.aaline(surface, (255, 0, 0), gap_start_inner, gap_start_outer)
            
            # Draw gap end line (green)
            gap_end_inner = self._get_point_on_circle_radius(gap_end, self.radius - self.width // 2 - 5)
            gap_end_outer = self._get_point_on_circle_radius(gap_end, self.radius + self.width // 2 + 5)
            pygame.draw.aaline(surface, (0, 255, 0), gap_end_inner, gap_end_outer)
            
            # Draw gap center line (yellow)
            gap_center = self._normalize_angle(gap_start + self.gap_angle / 2)
            gap_center_inner = self._get_point_on_circle_radius(gap_center, self.radius - self.width // 2 - 10)
            gap_center_outer = self._get_point_on_circle_radius(gap_center, self.radius + self.width // 2 + 10)
            pygame.draw.aaline(surface, (255, 255, 0), gap_center_inner, gap_center_outer)
    
    def set_rotation_speed(self, speed_degrees_per_sec):
        """Set the rotation speed in degrees per second."""
        self.rotation_speed = math.radians(speed_degrees_per_sec)


# Game state management
class GameState:
    def __init__(self, screen_width, screen_height, num_arcs=5, arc_spacing=15, 
                 base_radius=60, gap_angle=60, base_rotation_speed=90):
        """
        Initialize game state with customizable arc configuration.
        
        Args:
            screen_width, screen_height (int): Screen dimensions
            num_arcs (int): Number of concentric arcs to create
            arc_spacing (int): Spacing between arcs in pixels
            base_radius (int): Radius of the innermost arc
            gap_angle (float): Gap size in degrees for all arcs
            base_rotation_speed (float): Base rotation speed in degrees per second
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.num_arcs = num_arcs
        self.arc_spacing = arc_spacing
        self.base_radius = base_radius
        self.gap_angle = gap_angle
        self.base_rotation_speed = base_rotation_speed
        self.arcs = []  # List to hold all arcs
        self.reset_game()
    
    def reset_game(self):
        """Reset the game to initial state."""
        center = (self.screen_width // 2, self.screen_height // 2)
        
        # Create N arcs with customizable spacing (stored as stack - innermost last)
        self.arcs = []
        colors = [
            (255, 100, 100),  # Red
            (100, 255, 100),  # Green
            (100, 100, 255),  # Blue
            (255, 255, 100),  # Yellow
            (255, 100, 255),  # Magenta
            (100, 255, 255),  # Cyan
            (255, 150, 100),  # Orange
            (150, 100, 255),  # Purple
            (100, 255, 150),  # Light Green
            (255, 100, 150),  # Pink
        ]
        
        for i in range(self.num_arcs):
            # Calculate radius for this arc
            radius = self.base_radius + (i * self.arc_spacing)
            
            # Vary rotation speed and direction for each arc
            speed_multiplier = 1 + (i * 0.1)  # Each arc slightly faster
            rotation_speed = self.base_rotation_speed * speed_multiplier
            if i % 2 == 1:  # Alternate direction for odd-indexed arcs
                rotation_speed = -rotation_speed
            
            # Vary gap size slightly for variety
            gap_variation = self.gap_angle + (i * 0.5)  # Slightly larger gaps for outer arcs
            
            # Vary arc thickness
            arc_width = max(8, 20 - i)  # Inner arcs thicker, outer arcs thinner
            
            # Select color (cycle through available colors)
            color = colors[i % len(colors)]
            
            arc = RotatingArc(
                center=center,
                radius=radius,
                width=arc_width,
                color=color,
                gap_angle=gap_variation,
                rotation_speed=rotation_speed,
                segments=max(60, 80 - i * 5)  # Fewer segments for outer arcs (performance)
            )
            
            self.arcs.append(arc)
        
        # Create a ball at the center
        self.ball = Ball(center[0], center[1], radius=8, color=(255, 255, 0))
        
        self.escaped_arcs = 0
        self.total_arcs = len(self.arcs)
    
    def get_current_active_arc(self) -> RotatingArc:
        """Get the current innermost active arc (front of queue)."""
        return self.arcs[0] if self.arcs else None
    
    def escape_current_arc(self):
        """Remove the current active arc from the stack."""
        if self.arcs:
            escaped_arc = self.arcs.pop(0) # Remove the innermost arc (FIFO)
            escaped_arc.active = False
            self.escaped_arcs += 1
            print(f"Escaped through arc at radius {escaped_arc.radius}! ({self.escaped_arcs}/{self.total_arcs})")
            return True
        return False
    
    def set_arc_configuration(self, num_arcs=None, arc_spacing=None, base_radius=None, 
                            gap_angle=None, base_rotation_speed=None):
        """
        Update arc configuration and reset the game.
        
        Args:
            num_arcs (int): Number of arcs (if None, keep current)
            arc_spacing (int): Spacing between arcs (if None, keep current)
            base_radius (int): Base radius (if None, keep current)
            gap_angle (float): Gap angle (if None, keep current)
            base_rotation_speed (float): Base rotation speed (if None, keep current)
        """
        if num_arcs is not None:
            self.num_arcs = num_arcs
        if arc_spacing is not None:
            self.arc_spacing = arc_spacing
        if base_radius is not None:
            self.base_radius = base_radius
        if gap_angle is not None:
            self.gap_angle = gap_angle
        if base_rotation_speed is not None:
            self.base_rotation_speed = base_rotation_speed
        
        self.reset_game()


# Example usage:
if __name__ == "__main__":
    pygame.init()
    
    # Screen setup
    WIDTH, HEIGHT = 405, 720
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Ball Escape Challenge")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)

    video_recorder = VideoRecorder(fps=60, width=WIDTH, height=HEIGHT)
    video_recorder.start_recording()

    # Initialize game with customizable settings
    game = GameState(
        WIDTH, HEIGHT,
        num_arcs=20,          # Number of arcs
        arc_spacing=8,      # Narrow spacing between arcs
        base_radius=45,      # Starting radius
        gap_angle=30,        # Gap size in degrees
        base_rotation_speed=10  # Base rotation speed
    )
    
    running = True
    while running:
        # Calculate time delta in seconds
        dt = clock.tick(60) / 1000.0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    # Reset game
                    game.reset_game()
                elif event.key == pygame.K_SPACE:
                    # Pause/resume arcs
                    for arc in game.arcs:
                        current_speed = arc.rotation_speed
                        arc.set_rotation_speed(0 if current_speed != 0 else 90)
        
        # Update game objects
        game.ball.update(dt, WIDTH, HEIGHT, game.arcs)
        
        # Update game objects
        game.ball.update(dt, WIDTH, HEIGHT, game.arcs)
        
        # Update all arcs rotation
        for arc in game.arcs:
            arc.update(dt)
        
        # Check escape detection only for the current active arc (top of stack)
        current_arc = game.get_current_active_arc()
        if current_arc and current_arc.check_ball_escaped(game.ball):
            game.escape_current_arc()
        
        # Draw everything
        screen.fill((20, 20, 30))  # Dark background
        
        # Draw arcs (outermost first - reverse the stack for drawing)
        for arc in reversed(game.arcs):
            arc.draw(screen)
        
        # Draw ball
        game.ball.draw(screen)
        
        # Draw UI
        active_arcs = len(game.arcs)
        text = font.render("Est-ce que tu peux t'échapper ?", True, (255, 255, 255))
        if DEBUG:
            status_text = f"Active Arcs: {active_arcs}/{game.total_arcs} | Escaped: {game.escaped_arcs}"
            text = font.render(status_text, True, (200, 200, 200))

        screen.blit(text, (10, 10))
        
        # Show current target arc
        if game.arcs and DEBUG:
            current_arc = game.get_current_active_arc()
            target_text = f"Target Arc Radius: {current_arc.radius}"
            target_surface = font.render(target_text, True, (150, 255, 150))
            screen.blit(target_surface, (10, 50))

        if DEBUG:
            controls_text = font.render("R: Reset | SPACE: Pause | 1-4: Difficulty", True, (150, 150, 150))
            screen.blit(controls_text, (10, HEIGHT - 40))
        
        # Check win condition
        if active_arcs == 0:
            win_text = font.render("All arcs cleared! Press R to restart", True, (0, 255, 0))
            text_rect = win_text.get_rect(center=(WIDTH//2, HEIGHT//2))
            screen.blit(win_text, text_rect)
        
        pygame.display.flip()

        if video_recorder.is_recording():
            video_recorder.capture_frame(screen)
    
    if video_recorder.is_recording():
        video_recorder.stop_recording()
    
    pygame.quit()
