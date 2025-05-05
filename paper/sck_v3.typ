// Import required packages
#import "@preview/ctheorems:1.1.2": *
#import "@preview/showybox:2.0.1": showybox
#import "@preview/physica:0.9.4": *

// Define color palette
#let colors = (
  blue: rgb("#2196F3"),
  teal: rgb("#00BCD4"),
  red: rgb("#FF5722"),
  green: rgb("#72FC3F"),
  orange: rgb("#FFB84C"),
  orange_text: rgb("#FFB84C").darken(10%),
  pink: rgb("#F266AB"),
  violet: rgb("#A459D1"),
  turquoise: rgb("#2CD3E1"),
  lightgray: rgb("F5F5F5").lighten(25%)
)

#let niprod(a, b) = $lr(iprod(#a, #b), size: #(60% - 66pt).ratio)$

// Text formatting utilities
#let format_text(content, color: black, size: 8pt, weight: "regular") = {
  text(
    content,
    font: "Noto Sans Mono",
    fill: color,
    size: size,
    weight: weight
  )
}

#let raw_orange(content) = format_text(content, color: colors.orange)
#let raw_violet(content) = format_text(content, color: colors.violet)

// Theme components
#let theme_text(content, weight: "regular", color: rgb("#000000")) = {
  text(weight: weight, fill: color)[#content]
}

// Base theorem styling
#let create_theorem_base(
  identifier,
  head,
  color: rgb("#000000"),
  style: "box", // can be "box" or "line"
  shadow: none,
  bg_color: none,
  width: 50%
) = {
  let frame = if style == "box" {
    (
      body-color: if bg_color != none { bg_color } else { color.lighten(92%) },
      // border-color: color.darken(10%),
      border-color: color,
      thickness: 1.5pt,
      inset: 1.2em,
      radius: 0.3em,
    )
  } else {
    (
      body-color: if bg_color != none { bg_color } else { color.lighten(92%) },
      border-color: color.darken(10%),
      thickness: (left: 2pt),
      inset: 1.2em,
      radius: 0em,
    )
  }

  let base_style = thmenv(
    identifier,
    "heading",
    none,
    (name, number, body, title: auto, ..args) => {
      pad(
        top: 0.5em,
        bottom: 0.5em,
        showybox(
          width: width, 
          radius: 0.3em,
          breakable: true,
          padding: (top: 0em, bottom: 0em),
          frame: frame,
          shadow: shadow,
          [#body]
        )
      )
    }
  ).with(supplement: head, numbering: none)

  return base_style
}

// Predefined styles
#let problem = create_theorem_base(
  "",
  "Problem",
  color: colors.red,
  shadow: (offset: (x: 2pt, y: 2pt), color: luma(70%))
)

#let idea = create_theorem_base(
  "",
  "Idea",
  color: colors.blue,
  shadow: (offset: (x: 3pt, y: 3pt), color: luma(70%))
)

#let definition = create_theorem_base(
  "",
  "Definition",
  color: colors.teal,
  style: "line"
)

#let example = create_theorem_base(
  "",
  "Example",
  color: colors.red,
  style: "line"
)

// Conversation display function using theorem style
#let display_conversation(user_prompt:"", assistant_name:"", assistant_response:"", orange: true, size: 7pt, width: 50%) = {

  let bg_color = if orange { colors.orange.lighten(85%) } else { colors.violet.lighten(85%) }
  let border_color = if orange { colors.orange.darken(10%) } else { colors.violet.darken(10%) }
  
  let conversation_style = create_theorem_base(
    "",
    "Conversation",
    color: border_color,
    bg_color: bg_color,
    shadow: (offset: (x: 2pt, y: 2pt), color: luma(90%)), 
    width: width
  )

  //set text(size: 3pt)

  conversation_style[
    #format_text([
      #set text(size: size)
      #{if user_prompt != "" { 
        [*User:* #user_prompt] 
      }}
      
      #{if assistant_name != "" {
        [*#assistant_name:* ]
      }}
      
      #assistant_response
      
    ])
  ]
}
#let display_conversation_(user_prompt:"", assistant_name:none, assistant_response:none, c1: colors.orange, c2: colors.violet, grad: false, angle: 77deg, width: 50%, size: 7pt) = {
  let border_color = c1.darken(10%)
  let bg_color = c1.lighten(85%)

  if grad {
    border_color = gradient.linear(c1.darken(5%), c2.darken(5%), angle:angle)
    bg_color = gradient.linear(c1.lighten(85%), c2.lighten(85%), angle:angle)
  } 

  let conversation_style = create_theorem_base(
    "",
    "Conversation",
    color: border_color,
    bg_color: bg_color,
    shadow: (offset: (x: 2pt, y: 2pt), color: luma(90%)),
    width:width
  )

  //set text(size: 3pt)

  conversation_style[
    #format_text([
      #set text(size: size)
      #{if user_prompt != "" { 
        [*User:* #user_prompt] 
      }}
      
      #{if assistant_name != none {
        [*#assistant_name:* ]
      }}
      
      #{if assistant_response != none {
        [#assistant_response]
      }}
    ])
  ]
}




// Chat template components
#let chat_template(model_name: "qwen", system_prompt: "", user_prompt: "") = {
  let template = []
  
  let qwen_template = [
    <|im_start|>system \
    #text(fill: colors.orange_text)[#system_prompt] \
    <|im_end|><|im_start|>user \
    #text(fill: colors.violet)[#user_prompt]<|im_end|> \
    <|im_start|>assistant \
  ]
  let gemma_template = [
    \<bos>\<start_of_turn>user \
    #text(fill: colors.violet)[#user_prompt]\<end_of_turn> \
    \<start_of_turn>model
  ]
  let llama_template = [
    <|begin_of_text|><|start_header_id|>system<|end_header_id|> \

    #text(fill: colors.orange_text)[#system_prompt]
    
    <|eot_id|><|start_header_id|>user<|end_header_id|>

    #text(fill: colors.violet)[#user_prompt]<|eot_id|><|start_header_id|>assistant<|end_header_id|> \
  ]
  if model_name == "qwen" {
    template = qwen_template
  }
  if model_name == "gemma" {
    template = gemma_template
  }
  if model_name == "llama" {
    template = llama_template
  }
  set par(justify: false) 
  block(
    width: 100%,
    inset: 1em,
    fill: colors.lightgray,
    radius: 0.3em,
    text(
      font: "Noto Sans Mono",
      size: 7pt,
      template
    )
  )
}


#let contrast_data(harmful, harmless, before:[], separator:[], after:[]) = {
  let template = [
    #before
    #text(fill: colors.violet)[#harmful] \
    #separator
    #text(fill: colors.orange_text)[#harmless] \
    #after
  ]
  set par(justify: false) 
  block(
    width: 100%,
    inset: 1em,
    fill: colors.lightgray,
    radius: 0.3em,
    text(
      font: "Noto Sans Mono",
      size: 7pt,
      template
    )
  )
}
