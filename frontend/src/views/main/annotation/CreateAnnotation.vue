<template>
  <v-container fluid>
    <v-card class="ma-3 pa-3">
      <v-card-title primary-title>
        <div class="headline primary--text">Create Annotation</div>
      </v-card-title>
      <v-card-text>
        <AnnotationForm :annotation="newAnnotation"></AnnotationForm>
      </v-card-text>
      <v-card-actions>
        <v-spacer></v-spacer>
        <v-btn @click="cancel">Cancel</v-btn>
        <v-btn @click="reset">Reset</v-btn>
        <v-btn @click="submit"> Save </v-btn>
      </v-card-actions>
    </v-card>
  </v-container>
</template>

<script lang="ts">
import { Component, Vue } from 'vue-property-decorator';
import { Annotation, AnnotationUpdate, AnnotationCreate } from '@/api';
import { defaultAnnotation } from '@/interfaces';
import AnnotationForm from '@/components/AnnotationForm.vue';
import { dispatchCreateAnnotation } from '@/store/annotation/actions';
import { filterUndefined } from '@/utils';

@Component({ components: { AnnotationForm } })
export default class CreateAnnotation extends Vue {
  public newAnnotation: AnnotationCreate = defaultAnnotation();
  public valid = false;

  public async mounted() {
    this.reset();
  }

  public reset() {
    this.newAnnotation = defaultAnnotation();
    this.$validator.reset();
  }

  public cancel() {
    this.$router.back();
  }

  public async submit() {
    if (await this.$validator.validateAll()) {
      const filteredAnnotation: AnnotationCreate = filterUndefined(this.newAnnotation);
      await dispatchCreateAnnotation(this.$store, filteredAnnotation);
      this.$router.push('/main/annotations');
    }
  }
}
</script>
