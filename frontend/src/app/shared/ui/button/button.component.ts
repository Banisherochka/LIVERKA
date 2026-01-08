import { Component, HostBinding, Input } from '@angular/core';
import {NgClass} from '@angular/common';

type ButtonVariant = 'primary' | 'secondary' | 'ghost';
type ButtonSize = 'md' | 'lg';

@Component({
  selector: 'ui-button',
  standalone: true,
  templateUrl: './button.component.html',
  imports: [
    NgClass
  ],
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
}

