import { Component, HostBinding, Input } from '@angular/core';
import { NgClass } from '@angular/common';
import { MatButtonModule } from '@angular/material/button';

type ButtonVariant = 'primary' | 'secondary' | 'ghost';
type ButtonSize = 'md' | 'lg';

@Component({
  selector: 'ui-button',
  standalone: true,
  imports: [NgClass, MatButtonModule],
  templateUrl: './button.component.html',
  styleUrl: './button.component.scss'
})
export class ButtonComponent {
  @Input() type: 'button' | 'submit' = 'button';
  @Input() disabled = false;
  @Input() fullWidth = false;
  @Input() variant: ButtonVariant = 'primary';
  @Input() size: ButtonSize = 'md';

  @HostBinding('class.full-width')
  get hasFullWidth() {
    return this.fullWidth;
  }

  get classes() {
    return [`variant-${this.variant}`, `size-${this.size}`];
  }

  get matVariant(): 'raised' | 'flat' | 'stroked' | 'fab' | 'mini-fab' | 'icon' {
    switch (this.variant) {
      case 'primary': return 'raised';
      case 'secondary': return 'flat';
      case 'ghost': return 'stroked';
      default: return 'raised';
    }
  }
}

