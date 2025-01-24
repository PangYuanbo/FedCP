import SchemDraw
import SchemDraw.elements as elm

# Initialize drawing
with SchemDraw.Drawing() as d:
    # Power supply for the circuit (Vcc)
    d += elm.SourceV(label='Vcc')
    d.push()

    # Photocoupler (Optocoupler)
    d += elm.LED(label='Optocoupler LED')
    d += elm.Gap()  # Object detection area (shown as a gap)
    d += elm.Resistor().down().label('R1')
    d += elm.Ground()

    # Optocoupler control output
    d.pop()
    d += elm.Line().right()
    d += elm.Dot()  # Node
    d += elm.Line().down().length(2)
    d += elm.TransistorNpn(label='Q1', anchor='base')
    d += elm.Resistor().right().at((2, -1.5)).label('R2')  # Resistor connected to the transistor base
    d += elm.Line().up()
    d += elm.Dot()