
// Dimensionless Drag Coefficient * Cross-Sectional Area,
// Angle of Attack, and
// Mach speed Logger kOS Script

function main {
    clearscreen.
    print("Starting Dimensionless Drag Coefficient Logger...").
    print("Mach Speed, Angle of Attack, Cd * A").

    // Initialize variables
    declare local VESSEL_MASS to ship:mass.
    declare local specificGasConstant to 287.05. // Specific gas constant for air in J/(kg·K)
    
    // Open the log file
    log "Mach, AoA, Cd*A" to "log.log".

    // Loop until the vessel altitude is below 100 meters
    until (ship:altitude <= 100) {
        declare local vesselAltitude to ship:altitude.
        declare local surfaceVelocityMagnitude to ship:velocity:surface:mag.
        declare local machNumber to ship:mach.
        declare local angleOfAttack to ship:surface:aoa.

        // Retrieve atmospheric sensor data
        declare local pressure to vessel:sensors:pres.
        declare local temperature to vessel:sensors:temp.

        // Check if sensors are available
        if pressure > 0 and temperature > 0 {
            // Calculate atmospheric density using the Ideal Gas Law: rho = P / (R * T)
            declare local atmosphereDensity to pressure / (specificGasConstant * temperature).
            
            // Calculate gravitational force
            declare local gravitationalAcceleration to body:mu / (body:radius + vesselAltitude)^2.

            // Calculate drag force using acceleration
            declare local engineAcceleration to ship:maxthrust / VESSEL_MASS.
            declare local dragForce to (engineAcceleration - gravitationalAcceleration) * VESSEL_MASS.

            // Calculate Cd * A using the drag equation: Fd = 0.5 * ρ * v^2 * Cd * A
            if atmosphereDensity > 0 and surfaceVelocityMagnitude > 0 {
                declare local cdTimesArea to (2 * dragForce) / (atmosphereDensity * (surfaceVelocityMagnitude^2)).
                
                // Log data
                log (machNumber + ", " + angleOfAttack + ", " + cdTimesArea) to "log.log".
                
                print("Mach: " + machNumber + ", AoA: " + angleOfAttack + ", Cd*A: " + cdTimesArea).
            }
        }
        
        // Add a small wait to avoid spamming logs too quickly
        wait(0.1).
    }
}

main().
