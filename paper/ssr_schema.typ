#import "@preview/fletcher:0.5.4" as fletcher: diagram, node, edge
#import "sck_v3.typ": *
#import "@preview/cetz:0.3.2"

#let steering = cetz.canvas({
  import cetz.draw: *

    let s = 1.1
    let ax = s * 3
    let thicc = 1.5pt

    let g = (1, 2, 0).map(x => s*x) 
    let r1 = (0.5, -0.5, 0).map(x => s * x)
    let r2 = (1, 0, -0.5).map(x => s * x)  
    let y1 = (g.at(0) + r1.at(0), g.at(1) + r1.at(1), g.at(2) + r1.at(2))  
    let y2 = (y1.at(0) + r2.at(0), y1.at(1) + r2.at(1), y1.at(2) + r2.at(2))  
    
    set-style(mark: (end: "straight"))
    on-layer(1, {
      line((0,0,0), g, stroke: (paint: colors.orange, thickness: thicc), name: "g")
      line((0,0,0), y2, stroke: (paint: colors.violet, thickness: thicc), name: "y")
    })

    on-layer(-1, {
      line(g, y1, stroke: gray, name: "r1")
      content(
        ("r1.start", 50%, "r1.end"), 
        angle: "r1.end",
        padding: .1,
        anchor: "south",
        [$r_1$]
      )
      // line((0,0,0), y1, stroke: yellow, thickness: 2.0)
      line(y1, y2, stroke: gray, name: "r2")
      content(
        ("r2.start", 50%, "r2.end"), 
        angle: "r2.end",
        padding: .1,
        anchor: "south",
        [$r_2$]
      )
    })


    line((0, 0, 0), (ax, 0, 0), stroke: gray)
    line((0, 0, 0), (0, ax, 0), stroke: gray)
    // line((0, 0, 0), (0, 0, 3))
})


#set text(font: "Noto Sans Mono",  )

#let resid_width = 65pt 
#let add_width = 30pt
#let resid_height = 20pt
#let add_height = 25pt
#let radius_ = 3pt
#let neurone_radius = 15pt

#let blob(pos, label, tint: white, ..args) = node(
	pos, align(center, label),
	fill: tint.lighten(60%),
	stroke: 1pt + tint.darken(20%),
	corner-radius: radius_,
	..args,
)
#let blob_add(pos, label, tint: white, ..args) = blob(
  pos, 
  label, 
  tint: colors.orange, 
  width: add_width,
  height: add_height,
  ..args,
)
#let blob_resid(pos, label, tint: white, ..args) = blob(
  pos, 
  label, 
  tint: colors.orange, 
  width: resid_width,
  ..args,
)
#let neurone(pos, tint: colors.violet, ..args) = blob(
  pos, 
  [], 
  tint: tint, 
  width: neurone_radius,
  shape: circle,
  ..args,
)
#let probe_vertical = diagram(
  debug:1, 
  spacing: (15pt, 25pt),
  neurone((0, 1), name: <n1>),
  neurone((0, 2), name: <n2>),
  //neurone((0, 2), name: <n2>, tint: colors.orange, fill: gradient.linear(colors.orange, colors.violet)),
  neurone((0, 3), name: <n3>),
  neurone((0, 4), name: <n4>),
  neurone((0, 5), name: <n5>),
  
  neurone((1, 3), name: <sig>),

  edge(<n1>, <sig>),
  edge(<n2>, <sig>),
  edge(<n3>, <sig>),
  edge(<n4>, <sig>),
  edge(<n5>, <sig>),
)
#let probe = diagram(
  spacing: (15pt, 25pt),
  neurone((1, 0), name: <n1>),
  neurone((2, 0), name: <n2>),
  //neurone((0, 2), name: <n2>, tint: colors.orange, fill: gradient.linear(colors.orange, colors.violet)),
  neurone((3, 0), name: <n3>),
  neurone((4, 0), name: <n4>),
  neurone((5, 0), name: <n5>),
  
  neurone((3, -1), name: <sig>),

  edge(<n1>, <sig>),
  edge(<n2>, <sig>),
  edge(<n3>, <sig>),
  edge(<n4>, <sig>),
  edge(<n5>, <sig>),
)
#let atn_width = 15pt
#let atn_height = 15pt
#let atn(pos, fill: black) = blob(
  pos, 
  [$$], 
  tint: orange, 
  fill: fill, 
  corner-radius: 0pt, 
  stroke: black.transparentize(100%),
  width: atn_width, 
  height: atn_height
)
#let attn_pattern = diagram(
  spacing: (0pt, 0pt),
  atn((0,0), fill: colors.orange.transparentize(20%)),
  
  atn((0,1), fill: colors.orange.transparentize(80%)),
  atn((1,1), fill: colors.orange.transparentize(40%)),
  
  atn((0,2), fill: colors.orange.transparentize(65%)),
  atn((1,2), fill: colors.orange.transparentize(90%)),
  atn((2,2), fill: colors.orange.transparentize(65%)),
  
  atn((0,3), fill: colors.orange.transparentize(75%)),
  atn((1,3), fill: colors.orange.transparentize(90%)),
  atn((2,3), fill: colors.orange.transparentize(90%)),
  atn((3,3), fill: colors.orange.transparentize(65%)),
  
  atn((0,4), fill: colors.orange.transparentize(80%)),
  atn((1,4), fill: colors.orange.transparentize(95%)),
  atn((2,4), fill: colors.orange.transparentize(100%)),
  atn((3,4), fill: colors.violet.transparentize(30%)),
  atn((4,4), fill: colors.orange.transparentize(75%)),

  node((0, 0), [], name: <a>, width: atn_width - 10pt, height: atn_height - 10pt),
  node((4, 4), [], name: <b>, width: atn_width - 10pt, height: atn_height - 10pt),
  node([], enclose: (<a>, <b>), stroke: colors.orange + 1.5pt)
)

#let text_size = 10pt

#let steering_text = box(
  [
    #set text(size: text_size)
    #set par(justify: true)

    #par([*(S)* Instead of steering during inference, an adversarial suffix can be computed to trigger the same behavior, without the need of intervention.])
  ], 
  width: 120pt
)
#let probe_text = box(
  [
    #set text(size: text_size)
    #set par(justify: true)

    #par([*(P)* Once a probe has been trained, it can be used to backpropagate from anywhere inside the model.])
  ], 
  width: 120pt
)
#let attention_text = box(
  [
    #set text(size: text_size)
    #set par(justify: true)

    #par([*(A)* Optimisation can be done directly on the attention's pattern to smuggle a token under the targeted head's nose.]) 
  ], 
  width: 120pt
)


#let ssr_schema = diagram(
  debug: 0,
  spacing: (12pt, 20pt),
  
  node((-1, 5), steering, inset: -80pt, name: <probe>),
  node((-1, 7), steering_text),
  edge((-0.5, 5.5), (0, 5.5), "--"),
  edge((-0.8, 5.5), (-0.5, 5.5), "-->>"),

  node((4, 1), probe, inset: -80pt, name: <probe>),
  edge((4, 1.8), (4, 2.8), "-->>"),
  edge((4, 1.8), (2, 3.2), "--", corner: right, label: probe_text, label-sep: 0.8em, label-pos: 40%),
  node((2, 3.2), [], stroke: 1pt, name: <r3>, fill: black, radius: 1pt),
  
  node((13, 5.6), attn_pattern, inset: -80pt, name: <attn_pattern>),
  edge((3, 6.5), (9.6, 6.5), "--", label: attention_text, label-sep: 2em, label-side: right, label-pos: 100%),
  edge((6, 6.5), (6.1, 6.5), "<<--"),
  node((3, 6.5), [], stroke: 1pt, name: <ra>, fill: black, radius: 1pt),
 
  blob_resid((0,0.2), [logits], tint: red),
  edge("<|-"),
  blob_resid((0,1), [unembed], tint: red),
  edge("<|--/--", label: align(right, text("\n\nThe forward pass is\nstopped after the last\nintervention layer.", size: text_size)), label-sep: 1.5em),
  
  node((0,3), [+], stroke: 1pt, radius: 7pt, name: <r4>),
  edge("<|-"),
  node((0,5), [], stroke: 1pt, name: <r3>, fill: black, radius: 1pt),
  edge("-"),
  node((0,6), [+], stroke: 1pt, radius: 7pt, name: <r2>),
  edge("<|-"),
  node((0,8), [], stroke: 1pt, name: <r1>, fill: black, radius: 1pt),
  
  edge((0, 8), (0, 8.2), "<|-"),
  edge((0, 8), (0, 9), "--"),
  blob_resid((0, 9), [embed], tint: red),
  edge("<|-"),
  blob_resid((0,9.8), [tokens], tint: red),
  
  edge(<mlp>, <r4>, "-|>", corner: left),
  edge(<attn1>, <r2>, "-|>", corner: left),
  edge(<attn2>, <r2>, "-|>", corner: left),
  edge(<attn3>, <r2>, "-|>", corner: left),
  edge(<r3>, <mlp>, "-|>", corner: left),
  edge(<r1>, <attn1>, "-|>", corner: left),
  edge(<r1>, <attn2>, "-|>", corner: left),
  edge(<r1>, <attn3>, "-|>", corner: left),
  blob_add((1,7), [$h_1$], tint: orange, name: <attn1>),
  blob_add((2,7), [$dots$], tint: orange,name: <attn2>),
  blob_add((3,7), [$h_m$], tint: orange, name: <attn3>),

  node((1,4), [], name: <mlp1>, width: add_width, height: add_height),
  node((3,4), [], name: <mlp3>, width: add_width),
  node(
    [$"MLP" l$],
    enclose: (<mlp1>, <mlp3>), name: <mlp>,
    fill: colors.orange.lighten(60%),
    stroke: 1pt + colors.orange.darken(20%),
    corner-radius: radius_,
    inset: 0pt
  ),
)

#ssr_schema


#pagebreak()
