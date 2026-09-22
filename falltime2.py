import argparse
import numpy as np

def fall_with_drag(height: float, gravity: float, mass: float, drag_coeff: float, area: float, density: float):
    #function that takes height, gravity, mass, drag, area and density. i have made them all floats for my sanity
    
    #returns: (time_taken, final_velocity)
    
    # what to do if drag is 0
    drag_factor = 0.5 * density * drag_coeff * area # formula for drag 
    if drag_factor == 0:
        time_taken = np.sqrt((2 * height) / gravity)
        final_velocity = gravity * time_taken
        return time_taken, final_velocity

       
    dt = 0.001  # time stops in seconds
    time_taken = 0.0
    velocity = 0.0
    current_height = height

    while current_height > 0:
        # drag force: f_d = 0.5 * r * c_d * a * v^2
        drag_force = drag_factor * (velocity ** 2) #calculating drag force
        
        # gravitational force: f_g = m * g
        grav_force = mass * gravity
        
        # net acceleration: a = (f_g - f_d) / m
        acceleration = (grav_force - drag_force) / mass
        
        # velocity and position when height over 0
        velocity += acceleration * dt
        current_height -= velocity * dt
        time_taken += dt

    return time_taken, velocity

def main():
    parser = argparse.ArgumentParser(description="calculate drop time and final velocity with air resistance.")
    
    # the arguments that arg can parse haha
    parser.add_argument("-H", "--height", type=float, required=True, help="initial height from which the ball is dropped (meters)")
    
    # the fun stuff... gravity mass drag area
    #for each one i set a default so that the arguments are optional
    parser.add_argument("-g", "--gravity", type=float, default=9.81, help="acceleration due to gravity (default: 9.81 m/s^2).")

    parser.add_argument("-m", "--mass", type=float, default=0.520, help="mass of the ball in kilograms (default: 0.520 kg 520 sounds similar to i love you in mandarin (wo ai ni))" )

    parser.add_argument("-c", "--drag-coeff", type=float, default=0.3, help="drag coefficient cd of the ball (default: 0.3 for a smooth sphere, 0.0 for vacuum).")

    parser.add_argument( "-a", "--area", type=float, default=0.005, help="area in square meters, usually the cross-sectional area. i picked an arbitrary number... (default: 0.005 m^2)." )

    parser.add_argument("-rho", "--density", type=float, default=1.225, help="air density in kg/m^3 (default: 1.225 kg/m^3 at sea level).")
    
    args = parser.parse_args()
    
    # making sure the right values can input...
    if args.height <= 0:
        print("Error: Height must be greater than 0.")
        return
    if args.gravity <= 0 or args.mass <= 0 or args.drag_coeff < 0 or args.area < 0 or args.density < 0:
        print("Error: Physical constants must be positive numbers (or 0 for drag attributes).")
        return
        
    time_taken, final_velocity = fall_with_drag( args.height, args.gravity, args.mass, args.drag_coeff, args.area, args.density )
    
    # also calculating the terminal velocity
    drag_factor = 0.5 * args.density * args.drag_coeff * args.area
    print(f"\n--- fall results (with air resistance/drag) ---")
    print(f"drop height:       {args.height} m")
    print(f"gravity:           {args.gravity} m/s^2")
    print(f"object Mass:       {args.mass} kg")
    print(f"------------------------------------------")
    print(f"time to ground:    {time_taken:.4f} seconds")
    print(f"impact Velocity:   {final_velocity:.4f} m/s (or {final_velocity * 3.6:.2f} km/h)")
    
    if drag_factor > 0:
        v_terminal = np.sqrt((args.mass * args.gravity) / drag_factor)
        print(f"terminal velocity: {v_terminal:.4f} m/s (max speed possible for this object)")

if __name__ == "__main__":
    main()

